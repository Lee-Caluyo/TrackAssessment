from fastapi import FastAPI, HTTPException, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import threading
import time
import os
from datetime import datetime
import uuid
from sqlalchemy import create_engine, text

from routers.predict import router as rule_predict_router, compute_group_options
from careers_mapping import GROUP_CAREERS
from supabase_client import (
    insert_prediction,
    upsert_prediction,
    upsert_student_submission,
    get_all_results,
    insert_result,
    get_all_predictions,
    get_top_3_from_predictions,
    generate_result_id,
    update_prediction,
    fetch_msa_categories,
    fetch_msa_questions,
    fetch_msa_answers,
    get_user_result,
    get_latest_student_submission,
)
from supabase_client import SUPABASE_URL, SUPABASE_SERVICE_KEY
import requests
from urllib.parse import quote
import base64
import re
from fastapi import UploadFile, File

app = FastAPI()
app.include_router(rule_predict_router)

_extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"] + _extra_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,  # Ensure the allowed origins are passed here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
try:
    model = joblib.load("model/track_recommender.pkl")
except FileNotFoundError:
    print("Warning: model/track_recommender.pkl not found — continuing without RF model.")
except Exception as e:
    print(f"Warning: failed loading model: {e} — continuing without RF model.")

# Feature columns for the Random Forest model (must match training order)
MODEL_FEATURES = [
    "verbal_ability",
    "numerical_ability",
    "science_test",
    "clerical_ability",
    "interpersonal_skills_test",
    "logical_reasoning",
    "entrepreneurship_test",
    "mechanical_ability"
]


class StudentData(BaseModel):
    verbal_ability: int
    numerical_ability: int
    science_test: int
    clerical_ability: int
    interpersonal_skills_test: int
    logical_reasoning: int
    entrepreneurship_test: int
    mechanical_ability: int


def get_model_prediction(data: "StudentData"):
    """Get prediction from Random Forest model"""
    try:
        if model is None:
            return None, None
        # Prepare features in the correct order
        features = [[
            data.verbal_ability,
            data.numerical_ability,
            data.science_test,
            data.clerical_ability,
            data.interpersonal_skills_test,
            data.logical_reasoning,
            data.entrepreneurship_test,
            data.mechanical_ability
        ]]
        
        prediction = model.predict(features)[0]
       

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = max(proba) * 100
            return prediction, confidence
       
        return prediction, None
    except Exception as e:
        print(f"Model prediction error: {e}")
        return None, None


@app.post('/admin/fix_table_questions_sequence')
def admin_fix_table_questions_sequence(token: str = None):
    """Protected endpoint to fix the id sequence for Table_Questions.
    Call as: POST /admin/fix_table_questions_sequence?token=<ADMIN_FIX_TOKEN>
    The server reads `ADMIN_FIX_TOKEN` from env to authorize the call.
    """
    try:
        expected = os.getenv('ADMIN_FIX_TOKEN')
        if not expected:
            raise HTTPException(status_code=500, detail='ADMIN_FIX_TOKEN not configured on server')
        if not token or token != expected:
            raise HTTPException(status_code=403, detail='Invalid admin token')

        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise HTTPException(status_code=500, detail='DATABASE_URL not configured on server')

        engine = create_engine(database_url)
        with engine.begin() as conn:
            seq_name = conn.execute(text("SELECT pg_get_serial_sequence('public.Table_Questions','id') AS seqname")).scalar()
            conn.execute(text("SELECT setval(pg_get_serial_sequence('public.Table_Questions','id'), COALESCE((SELECT MAX(id) FROM public.Table_Questions), 0) + 1, false)"))
        return {"fixed_sequence": seq_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Track-related computation removed per user request.
# Track-related constants and compute functions were intentionally removed.

SYNC_INTERVAL_SECONDS = 5


def sync_predictions_from_results():
    """Sync predictions from student_submission table using new column names"""
    results = get_all_results()
    migrated_count = 0
    errors = []

    for row in results:
        try:
            # Map new column names from student_submission table
            verbal_ability = int(row.get("VerbalAbility", 0))
            numerical_ability = int(row.get("NumericalAbility", 0))
            science_test = int(row.get("ScienceTest", 0))
            clerical_ability = int(row.get("ClericalAbility", 0))
            interpersonal_skills_test = int(row.get("InterpersonalSkillsTest", 0))
            logical_reasoning = int(row.get("LogicalReasoning", 0))
            entrepreneurship_test = int(row.get("EntrepreneurshipTest", 0))
            mechanical_ability = int(row.get("MechanicalAbility", 0))

            # Calculate derived scores required by group computation
            va_et = (verbal_ability + entrepreneurship_test) / 2
            st_lr = (science_test + logical_reasoning) / 2

            scores = {
                "numerical_ability": numerical_ability,
                "clerical_ability": clerical_ability,
                "interpersonal_skills_test": interpersonal_skills_test,
                "mechanical_ability": mechanical_ability,
                "va_et": va_et,
                "st_lr": st_lr,
            }

            group_result = compute_group_options(scores, threshold=85)
            group_codes = group_result.get("groups", [])
            group_value = ", ".join(group_codes) if group_codes else None

            career_value = ", ".join([career for group in group_codes for career in GROUP_CAREERS.get(group, [])]) if group_codes else None

            # Build record matching student_submission schema (no Strand/track fields)
            record = {
                "group": group_value,
                "verbal_ability": float(verbal_ability),
                "numerical_ability": float(numerical_ability),
                "science_test": float(science_test),
                "clerical_ability": float(clerical_ability),
                "interpersonal_skills_test": float(interpersonal_skills_test),
                "logical_reasoning": float(logical_reasoning),
                "entrepreneurship_test": float(entrepreneurship_test),
                "mechanical_ability": float(mechanical_ability),
            }

            upsert_student_submission(record)
            migrated_count += 1
        except Exception as e:
            errors.append(f"Error processing row {row}: {str(e)}")

    return migrated_count, errors


@app.on_event("startup")
def startup_event():
    """Startup event. Automatic migration is disabled."""
    print("Startup: automatic migration from results to student_submission is disabled.")


def _background_sync():
    while True:
        try:
            sync_predictions_from_results()
        except Exception as e:
            print(f"Background sync failed: {str(e)}")
        time.sleep(SYNC_INTERVAL_SECONDS)



@app.get("/")
def root():
    return {"message": "API running 🚀"}


@app.get("/predictions")
def get_predictions():
    try:
        predictions = get_all_predictions()
        if not predictions:
            sync_predictions_from_results()
            predictions = get_all_predictions()
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


@app.post("/recompute-predictions")
def recompute_predictions():
    try:
        predictions = get_all_predictions()
        updated = 0
        errors = []

        for row in predictions:
            try:
                prediction_id = row.get("id")
                if not prediction_id:
                    continue

                numerical_ability = float(row.get("numerical_ability") or 0)
                clerical_ability = float(row.get("clerical_ability") or 0)
                interpersonal_skills_test = float(row.get("interpersonal_skills_test") or 0)
                mechanical_ability = float(row.get("mechanical_ability") or 0)
                va_et = float(row.get("va_et") or 0)
                st_lr = float(row.get("st_lr") or 0)

                scores = {
                    "numerical_ability": numerical_ability,
                    "clerical_ability": clerical_ability,
                    "interpersonal_skills_test": interpersonal_skills_test,
                    "mechanical_ability": mechanical_ability,
                    "va_et": va_et,
                    "st_lr": st_lr,
                }

                group_result = compute_group_options(scores, threshold=85)
                group_codes = group_result.get("groups", [])
                group_value = ", ".join(group_codes) if group_codes else None

                career_value = ", ".join([
                    career for group in group_codes for career in GROUP_CAREERS.get(group, [])
                ]) if group_codes else None

                update_prediction(
                    prediction_id,
                    {
                        "group": group_value,
                        "career": career_value,
                    },
                )
                updated += 1
            except Exception as e:
                errors.append(str(e))

        return {"updated": updated, "errors": errors}
    except Exception as e:
        return {"error": str(e)}


@app.get("/top-3-analysis")
def get_top_3_analysis():
    try:
        analysis = get_top_3_from_predictions()
        return analysis
    except Exception as e:
        return {"error": str(e)}


@app.post("/insert-result")
def insert_result_endpoint(data: StudentData):
    try:
        result_id = generate_result_id()
        record = {
            "id": result_id,
            "verbal_ability": data.verbal_ability,
            "numerical_ability": data.numerical_ability,
            "science_test": data.science_test,
            "clerical_ability": data.clerical_ability,
            "interpersonal_skills_test": data.interpersonal_skills_test,
            "logical_reasoning": data.logical_reasoning,
            "entrepreneurship_test": data.entrepreneurship_test,
            "mechanical_ability": data.mechanical_ability,
        }

        insert_result(record)
        return {"message": "Submission inserted successfully", "result_id": result_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/migrate")
def migrate_data():
    try:
        migrated_count, errors = sync_predictions_from_results()
        return {
            "message": f"Successfully migrated {migrated_count} records from results to student_submission",
            "errors": errors,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict_track(data: StudentData):
    try:
        scores = {
            "numerical_ability": data.numerical_ability,
            "clerical_ability": data.clerical_ability,
            "interpersonal_skills_test": data.interpersonal_skills_test,
            "mechanical_ability": data.mechanical_ability,
        }
        # Compute group-based mapping
        group_result = compute_group_options(scores, threshold=85)
        group_codes = group_result.get("groups", [])
        group_value = ", ".join(group_codes) if group_codes else None

        # Get Random Forest model prediction (kept for compatibility)
        rf_prediction, rf_confidence = get_model_prediction(data)

        record = {
            "id": generate_result_id(),
            "verbal_ability": data.verbal_ability,
            "numerical_ability": data.numerical_ability,
            "science_test": data.science_test,
            "clerical_ability": data.clerical_ability,
            "interpersonal_skills_test": data.interpersonal_skills_test,
            "logical_reasoning": data.logical_reasoning,
            "entrepreneurship_test": data.entrepreneurship_test,
            "mechanical_ability": data.mechanical_ability,
        }

        try:
            insert_result(record)
            db_status = "Inserted via REST ✅"
        except Exception as e:
            db_status = f"Insert failed ❌: {str(e)}"

        return {
            "group_based_result": group_result,
            "rf_model_prediction": str(rf_prediction) if rf_prediction else None,
            "rf_confidence": round(rf_confidence, 2) if rf_confidence else None,
            "db_status": db_status,
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


# ---------------------------------------------------------------------------
# MSA category code → student_submission column name
# ---------------------------------------------------------------------------
MSA_CODE_TO_FIELD = {
    "VA": "verbal_ability",
    "NA": "numerical_ability",
    "ST": "science_test",
    "CA": "clerical_ability",
    "IST": "interpersonal_skills_test",
    "ET": "entrepreneurship_test",
    "LR": "logical_reasoning",
    "MA": "mechanical_ability",
}

CATEGORY_PREFIX_OVERRIDES = {
    'verbal ability': 'VA',
    'numerical ability': 'NA',
    'science test': 'ST',
    'clerical ability': 'CA',
    'interpersonal skills test': 'IST',
    'entrepreneurship test': 'ET',
    'logical reasoning': 'LR',
    'mechanical ability': 'MA',
}

UPLOAD_ASSET_NAME_MAP = {
    "question": "Question",
    "opt_a": "OPT-A",
    "opt_b": "OPT-B",
    "opt_c": "OPT-C",
    "opt_d": "OPT-D",
    "opt_e": "OPT-E",
}


def prefix_from_category(cat: str) -> str:
    """Return standardized prefix for a human-friendly category name.
    Uses `CATEGORY_PREFIX_OVERRIDES` when available, otherwise falls back to
    taking the first letter of each word and uppercasing them.
    """
    if not cat:
        return ''
    key = cat.strip().lower()
    if key in CATEGORY_PREFIX_OVERRIDES:
        return CATEGORY_PREFIX_OVERRIDES[key]
    # fallback: initials of words
    return ''.join([w[0].upper() for w in cat.split() if w])


class ComputeFromSupabaseRequest(BaseModel):
    user_id: str


@app.post("/compute-from-supabase")
def compute_from_supabase(request: ComputeFromSupabaseRequest):
    """Fetch a user's MSA answers from Supabase, compute scores, and upsert into student_submission."""
    try:
        user_id = request.user_id

        # -- Fetch reference data --
        msa_categories = fetch_msa_categories()
        msa_questions = fetch_msa_questions()

        # Build lookup maps
        # right_answer keyed by (category_id, question_number)
        right_answers = {
            (q["category_id"], q["question_number"]): q["right_answer"]
            for q in msa_questions
        }
        # total questions per category for normalisation
        questions_per_msa_cat: dict = {}
        for q in msa_questions:
            cat_id = q["category_id"]
            questions_per_msa_cat[cat_id] = questions_per_msa_cat.get(cat_id, 0) + 1

        # -- Grade MSA answers --
        msa_answers = fetch_msa_answers(user_id)
        correct_per_cat: dict = {}
        for ans in msa_answers:
            cat_id = ans["category_id"]
            q_num = ans["question_number"]
            correct = right_answers.get((cat_id, q_num))
            if correct is not None and ans["answer"] == correct:
                correct_per_cat[cat_id] = correct_per_cat.get(cat_id, 0) + 1

        # Compute ability score (0–5 scale: correct / total * 5)
        ability_scores: dict = {}
        for cat in msa_categories:
            cat_id = cat["id"]
            code = cat["code"]
            field = MSA_CODE_TO_FIELD.get(code)
            if field is None:
                continue
            total = questions_per_msa_cat.get(cat_id, 0)
            if total == 0:
                ability_scores[field] = 0.0
                continue
            correct = correct_per_cat.get(cat_id, 0)
            ability_scores[field] = (correct / total) * 5


        # -- Build and upsert record (abilities) --
        verbal_ability = ability_scores.get("verbal_ability", 0)
        numerical_ability = ability_scores.get("numerical_ability", 0)
        science_test = ability_scores.get("science_test", 0)
        clerical_ability = ability_scores.get("clerical_ability", 0)
        interpersonal_skills_test = ability_scores.get("interpersonal_skills_test", 0)
        entrepreneurship_test = ability_scores.get("entrepreneurship_test", 0)
        logical_reasoning = ability_scores.get("logical_reasoning", 0)
        mechanical_ability = ability_scores.get("mechanical_ability", 0)

        record = {
            "user_id": user_id,
            "verbal_ability": float(verbal_ability),
            "numerical_ability": float(numerical_ability),
            "science_test": float(science_test),
            "clerical_ability": float(clerical_ability),
            "interpersonal_skills_test": float(interpersonal_skills_test),
            "logical_reasoning": float(logical_reasoning),
            "entrepreneurship_test": float(entrepreneurship_test),
            "mechanical_ability": float(mechanical_ability),
        }

        try:
            upsert_student_submission(record, conflict_column="user_id")
        except Exception as upsert_err:
            raise RuntimeError(f"Failed to upsert student_submission for user_id={user_id}: {upsert_err}") from upsert_err

        return {"message": "Computed and upserted successfully", "result": record}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/user-result/{user_id}")
def get_user_result_endpoint(user_id: str):
    """Fetch the student_submission record for a given user_id."""
    try:
        result = get_user_result(user_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No result found for user_id: {user_id}")
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.get("/student-result/{user_id}")
def get_student_result(user_id: str):
    """
    Get the latest assessment result for a specific user from student_submission table.
    Returns MSA abilities and track score fields.
    """
    try:
        result = get_latest_student_submission(user_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No result found for user_id: {user_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.get("/verbal-questions")
def get_verbal_questions():
    """Proxy endpoint that returns rows from the Supabase table "Verbal Ability".
    Uses SUPABASE_URL and SUPABASE_ANON_KEY environment variables.
    """
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        anon = os.getenv("SUPABASE_ANON_KEY")
        service_key = os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url:
            return {"error": "Supabase URL not configured on server"}
        # Prefer anon key for public access; fall back to service key if anon missing
        key_to_use = anon or service_key
        if not key_to_use:
            return {"error": "Supabase anon or service key not configured on server"}

        # Try a few candidate table names (some projects use different table names)
        candidates = ['"Verbal Ability"', 'Questionnaire_verbal_ability', 'verbal_ability', 'public."Verbal Ability"', 'public.Questionnaire_verbal_ability', 'public.verbal_ability']
        headers = {
            "apikey": key_to_use,
            "Authorization": f"Bearer {key_to_use}",
            "Accept": "application/json",
        }

        last_err = None
        for cand in candidates:
            table = quote(cand)
            url = f"{supabase_url}/rest/v1/{table}?select=id,question_id,question,A,B,C,D,answer,long_text,created_at,created_by&order=question_id.asc"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            last_err = {"status": resp.status_code, "text": resp.text}

        return {"error": "Supabase error: could not find table", "detail": last_err}
    except Exception as e:
        return {"error": str(e)}


@app.get("/questions")
def get_questions(category: str):
    """Generic proxy for questionnaire tables. `category` should be a human-friendly name
    like 'Verbal Ability' or 'Numerical Ability'. The endpoint will try a list of
    candidate table names in Supabase and return the first successful result.
    """
    try:
        supabase_url = os.getenv("SUPABASE_URL") or SUPABASE_URL
        anon = os.getenv("SUPABASE_ANON_KEY")
        service_key = os.getenv("SUPABASE_SERVICE_KEY") or SUPABASE_SERVICE_KEY
        if not supabase_url:
            return {"error": "Supabase URL not configured on server"}
        key_to_use = anon or service_key
        if not key_to_use:
            return {"error": "Supabase anon or service key not configured on server"}

        headers = {
            "apikey": key_to_use,
            "Authorization": f"Bearer {key_to_use}",
            "Accept": "application/json",
        }

        # Use single table `Table_Questions` and filter by question_id prefix derived from category
        prefix = prefix_from_category(category)
        table = quote('Table_Questions')
        # The table columns use opt_a..opt_e instead of A..D
        select_fields = 'id,question_id,question,long_text,image_url,opt_a,opt_b,opt_c,opt_d,opt_e,answer,created_at,created_by,uuid'        # If no prefix provided, return all rows (beware large result sets)
        if prefix:
            url = f"{supabase_url}/rest/v1/{table}?select={select_fields}&question_id=like.{prefix}%25&order=question_id.asc"
        else:
            url = f"{supabase_url}/rest/v1/{table}?select={select_fields}&order=question_id.asc"

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return {"error": "Supabase error", "status": resp.status_code, "text": resp.text}
        except Exception as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/questions")
def create_question(category: str, payload: dict):
    """Insert a new question row into the Supabase questionnaire table for `category`.
    Expects JSON payload with columns appropriate for the table (e.g. question_id, question, A,B,C,D,answer).
    """
    try:
        supabase_url = os.getenv("SUPABASE_URL") or SUPABASE_URL
        key = os.getenv("SUPABASE_SERVICE_KEY") or SUPABASE_SERVICE_KEY
        anon = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not (key or anon):
            return {"error": "Supabase URL or keys not configured"}
        headers = {
            "apikey": key or anon,
            "Authorization": f"Bearer {key or anon}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        # Map incoming payload fields to the Table_Questions schema
        def _map_payload(p: dict) -> dict:
            mapped = {}
            # pass through known fields
            for k in ('question_id', 'question', 'long_text', 'created_by'):
                if k in p:
                    mapped[k] = p[k]
            # allow passing an image URL for the question (storage public URL)
            if 'image_url' in p:
                mapped['image_url'] = p.get('image_url')
            # map options A..E or opt_a..opt_e
            if 'opt_a' in p or 'A' in p:
                mapped['opt_a'] = p.get('opt_a') or p.get('A')
            if 'opt_b' in p or 'B' in p:
                mapped['opt_b'] = p.get('opt_b') or p.get('B')
            if 'opt_c' in p or 'C' in p:
                mapped['opt_c'] = p.get('opt_c') or p.get('C')
            if 'opt_d' in p or 'D' in p:
                mapped['opt_d'] = p.get('opt_d') or p.get('D')
            # optional fifth option
            mapped['opt_e'] = p.get('opt_e') or p.get('E') or None
            if 'answer' in p:
                mapped['answer'] = p['answer']
            return mapped

        mapped_payload = _map_payload(payload)

        # Ensure question_id prefix matches category (frontend may generate question_id)
        prefix = prefix_from_category(category)
        qid = mapped_payload.get('question_id')
        # If no question_id provided, compute next available id with the category prefix
        if prefix and (not qid or str(qid).strip() == ''):
            try:
                seq_url = f"{supabase_url}/rest/v1/{quote('Table_Questions')}?select=question_id&question_id=like.{prefix}%25"
                seq_resp = requests.get(seq_url, headers=headers, timeout=8)
                if seq_resp.status_code == 200:
                    items = seq_resp.json()
                    nums = []
                    for it in items:
                        val = (it.get('question_id') or '')
                        if not isinstance(val, str):
                            val = str(val)
                        m = None
                        try:
                            import re
                            m = re.match(r'^' + re.escape(prefix) + r'(\d+)$', val)
                        except Exception:
                            m = None
                        if m:
                            try:
                                nums.append(int(m.group(1)))
                            except Exception:
                                pass
                    # Find the smallest missing positive integer (fill gaps)
                    if nums:
                        nums_set = set(nums)
                        next_num = 1
                        while next_num in nums_set:
                            next_num += 1
                    else:
                        next_num = 1
                    mapped_payload['question_id'] = f"{prefix}{str(next_num).zfill(3)}"
                else:
                    # fallback: use simple numeric id
                    mapped_payload['question_id'] = f"{prefix}001"
            except Exception:
                mapped_payload['question_id'] = f"{prefix}001"
        elif prefix and qid and not qid.startswith(prefix):
            # enforce prefix if provided but missing
            mapped_payload['question_id'] = f"{prefix}{qid}"

        # Ensure a uuid exists for public identification; server-generate if missing
        if 'uuid' not in mapped_payload or not mapped_payload.get('uuid'):
            try:
                mapped_payload['uuid'] = str(uuid.uuid4())
            except Exception:
                mapped_payload['uuid'] = None

        table = quote('Table_Questions')

        # If configured, perform an atomic direct-DB insert that finds the smallest
        # missing positive `id` and uses it. This fills gaps (e.g., missing 17 in 1..30)
        # and avoids sequence-based duplicate-key errors. Requires DATABASE_URL and
        # USE_DIRECT_DB_INSERT=1 in env. Falls back to REST insertion on failure.
        use_direct = os.getenv('USE_DIRECT_DB_INSERT', '').strip() == '1'
        database_url = os.getenv('DATABASE_URL')
        if use_direct and database_url:
            try:
                engine = create_engine(database_url)
                insert_sql = text(
                    "WITH next_id AS ("
                    "  SELECT n FROM (SELECT generate_series(1, COALESCE((SELECT MAX(id) FROM public.Table_Questions),0)+1) AS n) g"
                    "  LEFT JOIN public.Table_Questions t ON t.id = g.n"
                    "  WHERE t.id IS NULL ORDER BY n LIMIT 1"
                    ")"
                    "INSERT INTO public.Table_Questions (id, question_id, question, long_text, opt_a, opt_b, opt_c, opt_d, opt_e, answer, created_by, image_url, uuid)"
                    "SELECT next_id.n, :question_id, :question, :long_text, :opt_a, :opt_b, :opt_c, :opt_d, :opt_e, :answer, :created_by, :image_url, COALESCE(:uuid, gen_random_uuid()::text)"
                    "FROM next_id RETURNING *;"
                )

                params = {
                    'question_id': mapped_payload.get('question_id'),
                    'question': mapped_payload.get('question'),
                    'long_text': mapped_payload.get('long_text'),
                    'opt_a': mapped_payload.get('opt_a'),
                    'opt_b': mapped_payload.get('opt_b'),
                    'opt_c': mapped_payload.get('opt_c'),
                    'opt_d': mapped_payload.get('opt_d'),
                    'opt_e': mapped_payload.get('opt_e'),
                    'answer': mapped_payload.get('answer'),
                    'created_by': mapped_payload.get('created_by'),
                    'image_url': mapped_payload.get('image_url'),
                    'uuid': mapped_payload.get('uuid'),
                }

                with engine.begin() as conn:
                    res = conn.execute(insert_sql, params)
                    row = res.fetchone()
                    if row:
                        # convert Row to dict
                        data = dict(row._mapping)
                        # Advance the sequence to MAX(id)+1 to keep it ahead
                        try:
                            conn.execute(text(
                                "SELECT setval(pg_get_serial_sequence('public.\"Table_Questions\"','id'), COALESCE((SELECT MAX(id) FROM public.\"Table_Questions\"), 0) + 1, false)"
                            ))
                        except Exception:
                            # if sequence update fails, continue; sequence can be fixed via admin endpoint
                            pass
                        import json
                        return Response(content=json.dumps([data]), status_code=201, media_type="application/json")
            except Exception as e:
                print(f"Direct DB insert failed, falling back to REST: {e}")

        # Fallback: use Supabase REST insert (existing behavior)
        url = f"{supabase_url}/rest/v1/{table}"
        try:
            print(f"create_question: attempting POST to {url} payload={mapped_payload}")
            resp = requests.post(url, json=mapped_payload, headers=headers, timeout=10)
            resp_text = resp.text if hasattr(resp, 'text') else '<no body>'
            print(f"create_question: status={resp.status_code} resp={resp_text}")
            if resp.status_code in (200, 201):
                return Response(content=resp.text, status_code=resp.status_code, media_type="application/json")
            # forward error status and body
            raise HTTPException(status_code=resp.status_code, detail=resp_text)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        return {"error": str(e)}


@app.patch("/questions/{row_id}")
def update_question(row_id: str, category: str, payload: dict):
    """Update a question row by id in the Supabase table for `category`."""
    try:
        supabase_url = os.getenv("SUPABASE_URL") or SUPABASE_URL
        key = os.getenv("SUPABASE_SERVICE_KEY") or SUPABASE_SERVICE_KEY
        anon = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not (key or anon):
            return {"error": "Supabase URL or keys not configured on server"}

        headers = {
            "apikey": key or anon,
            "Authorization": f"Bearer {key or anon}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        mapped = {}
        for k in ('question_id', 'question', 'long_text', 'created_by', 'image_url'):
            if k in payload:
                mapped[k] = payload[k]

        if 'opt_a' in payload or 'A' in payload:
            mapped['opt_a'] = payload.get('opt_a') or payload.get('A')
        if 'opt_b' in payload or 'B' in payload:
            mapped['opt_b'] = payload.get('opt_b') or payload.get('B')
        if 'opt_c' in payload or 'C' in payload:
            mapped['opt_c'] = payload.get('opt_c') or payload.get('C')
        if 'opt_d' in payload or 'D' in payload:
            mapped['opt_d'] = payload.get('opt_d') or payload.get('D')

        mapped['opt_e'] = payload.get('opt_e') or payload.get('E') or None

        if 'answer' in payload:
            mapped['answer'] = payload['answer']

        table = quote('Table_Questions')

        is_int = False
        is_uuid = False
        try:
            int(row_id)
            is_int = True
        except Exception:
            is_int = False

        try:
            uuid.UUID(row_id)
            is_uuid = True
        except Exception:
            is_uuid = False

        if is_int:
            url = f"{supabase_url}/rest/v1/{table}?id=eq.{quote(row_id)}"
        elif is_uuid:
            url = f"{supabase_url}/rest/v1/{table}?uuid=eq.{quote(row_id)}"
        else:
            url = f"{supabase_url}/rest/v1/{table}?question_id=eq.{quote(row_id)}"

        try:
            print(f"update_question: PATCH -> {url} payload={mapped}")
        except Exception:
            pass

        resp = requests.patch(url, json=mapped, headers=headers, timeout=10)

        try:
            print(f"update_question: supabase status={resp.status_code} body={getattr(resp, 'text', '<no body>')}")
        except Exception:
            pass

        if resp.status_code in (200, 204):
            try:
                return Response(content=resp.text, status_code=200, media_type="application/json")
            except Exception:
                return {"updated": True}

        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}
@app.get("/user-profile")
def get_user_profile(email: str):
    """Lookup a users_profile row by email and return it.
    Expects a Supabase table named `users_profile` with an `email` and `usertype` column.
    """
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

        # safe-encode email for URL
        q = quote(email)
        # Query the `users_profile` table (note plural `users_profile`)
        url = f"{supabase_url}/rest/v1/users_profile?email=eq.{q}&select=*&limit=1"
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {"error": "Supabase error", "status": resp.status_code, "detail": resp.text}
        data = resp.json()
        if data:
            user = data[0]
            if 'email' in user and isinstance(user['email'], str):
                user['email'] = user['email'].strip()
            if 'password' in user:
                user.pop('password', None)
            return user

        # Fallback: try a case-insensitive / wildcard match in case stored emails have
        # trailing whitespace or newline characters (some rows have '\r\n').
        try:
            ilike_val = quote(f"%{email}%")
            alt_url = f"{supabase_url}/rest/v1/users_profile?email=ilike.{ilike_val}&select=*&limit=1"
            resp2 = requests.get(alt_url, headers=headers, timeout=8)
            if resp2.status_code == 200:
                data2 = resp2.json()
                if data2:
                    user = data2[0]
                    if 'email' in user and isinstance(user['email'], str):
                        user['email'] = user['email'].strip()
                    if 'password' in user:
                        user.pop('password', None)
                    return user
        except Exception:
            pass

        return {}
    except Exception as e:
        return {"error": str(e)}




def _decode_data_url(data_url: str):
    """Decode a data URL (data:[<mediatype>][;base64],<data>) into (bytes, content_type)"""
    m = re.match(r'^data:(?P<ctype>[^;]+)(;base64)?,(?P<data>.*)$', data_url, re.DOTALL)
    if not m:
        raise ValueError('Invalid data URL')
    ctype = m.group('ctype')
    data = m.group('data')
    if ';base64' in data_url:
        raw = base64.b64decode(data)
    else:
        raw = data.encode('utf-8')
    return raw, ctype


@app.post('/upload_dataurl')
def upload_dataurl(payload: dict):
    """Upload a single data URL to Supabase storage and return a public URL."""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail='Supabase storage not configured on server')

        bucket = payload.get('bucket')
        path = payload.get('path')
        data_url = payload.get('data_url')

        if not bucket or not path or not data_url:
            raise HTTPException(status_code=400, detail='bucket, path and data_url required')

        raw, ctype = _decode_data_url(data_url)

        upload_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{quote(path, safe='/')}"
        params = {'upsert': 'true', 'cacheControl': '3600'}
        headers = {
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'apikey': SUPABASE_SERVICE_KEY,
            'Content-Type': ctype,
        }

        resp = requests.put(upload_url, params=params, headers=headers, data=raw, timeout=20)
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=502, detail=f'Upload failed: {resp.status_code} {resp.text}')

        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{quote(path, safe='/')}"
        return {'public_url': public_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-question-image")
async def upload_question_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    asset_type: str = Form(...),
    question_id: str = Form(...),
    staging: str = Form("0"),
):
    
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail="Supabase storage not configured on server")

        category = (category or "").strip()
        asset_type = (asset_type or "").strip().lower()
        question_id = (question_id or "").strip().upper()
        is_staging = str(staging or "").strip().lower() in ("1", "true", "yes")

        prefix = prefix_from_category(category)
        if not prefix:
            raise HTTPException(status_code=400, detail="Invalid category")

        if asset_type not in UPLOAD_ASSET_NAME_MAP:
            raise HTTPException(status_code=400, detail="Invalid asset_type")

        if not question_id:
            raise HTTPException(status_code=400, detail="Missing question_id")

        if not question_id.startswith(prefix):
            raise HTTPException(status_code=400, detail="question_id does not match category")

        bucket = "msa-images"

        asset_label = UPLOAD_ASSET_NAME_MAP.get(asset_type)
        if not asset_label:
            raise HTTPException(status_code=400, detail="Invalid asset_type")

        filename = f"{prefix}-{asset_label}.webp"
        if is_staging:
            folder_name = "Temp"
            storage_path = f"Temp/{question_id} {filename}"
        else:
            folder_name = question_id
            storage_path = f"{folder_name}/{filename}"

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        upload_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{quote(storage_path, safe='/')}"
        upload_headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "image/webp",
        }
        params = {
            "upsert": "true",
            "cacheControl": "3600",
        }

        resp = requests.put(
            upload_url,
            params=params,
            headers=upload_headers,
            data=content,
            timeout=30,
        )

        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"Upload failed: {resp.status_code} {resp.text}"
            )

        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{quote(storage_path, safe='/')}"

        return {
            "success": True,
            "bucket": bucket,
            "folder": folder_name,
            "path": storage_path,
            "filename": filename,
            "public_url": public_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/questions/{row_id}')
def delete_question(row_id: str, category: str = ''):
    try:
        supabase_url = os.getenv('SUPABASE_URL') or SUPABASE_URL
        key = os.getenv('SUPABASE_SERVICE_KEY') or SUPABASE_SERVICE_KEY
        anon = os.getenv('SUPABASE_ANON_KEY')

        if not supabase_url or not (key or anon):
            return {"error": "Supabase URL or keys not configured"}

        auth_key = key or anon

        headers = {
            "apikey": auth_key,
            "Authorization": f"Bearer {auth_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        table = quote('Table_Questions')

        is_int = False
        is_uuid = False

        try:
            int(row_id)
            is_int = True
        except Exception:
            is_int = False

        try:
            uuid.UUID(row_id)
            is_uuid = True
        except Exception:
            is_uuid = False

        if is_int:
            lookup_url = f"{supabase_url}/rest/v1/{table}?id=eq.{quote(str(row_id))}&select=id,question_id,image_url,opt_a,opt_b,opt_c,opt_d,opt_e,uuid"
        elif is_uuid:
            lookup_url = f"{supabase_url}/rest/v1/{table}?uuid=eq.{quote(str(row_id))}&select=id,question_id,image_url,opt_a,opt_b,opt_c,opt_d,opt_e,uuid"
        else:
            lookup_url = f"{supabase_url}/rest/v1/{table}?question_id=eq.{quote(str(row_id))}&select=id,question_id,image_url,opt_a,opt_b,opt_c,opt_d,opt_e,uuid"

        lookup_resp = requests.get(lookup_url, headers=headers, timeout=10)

        if lookup_resp.status_code != 200:
            return {
                "error": "Lookup failed before delete",
                "status": lookup_resp.status_code,
                "text": lookup_resp.text
            }

        rows = lookup_resp.json()
        if not rows:
            return {
                "error": "No matching row found",
                "deleted": False,
                "row_id": row_id
            }

        row = rows[0]
        actual_id = row.get("id")
        question_id = (row.get("question_id") or "").strip().upper()

        if not actual_id:
            return {
                "error": "Matched row is missing numeric id",
                "deleted": False,
                "row": row
            }

        storage_deleted = []
        storage_errors = []

        if key and question_id:
            bucket = "msa-images"
            prefix = prefix_from_category(category) or re.sub(r'\d+$', '', question_id).upper()

            candidate_paths = [
                f"{question_id}/{prefix}-Question.webp",
                f"{question_id}/{prefix}-OPT-A.webp",
                f"{question_id}/{prefix}-OPT-B.webp",
                f"{question_id}/{prefix}-OPT-C.webp",
                f"{question_id}/{prefix}-OPT-D.webp",
                f"{question_id}/{prefix}-OPT-E.webp",
                f"{question_id}/{prefix}-Question.jpg",
                f"{question_id}/{prefix}-OPT-A.jpg",
                f"{question_id}/{prefix}-OPT-B.jpg",
                f"{question_id}/{prefix}-OPT-C.jpg",
                f"{question_id}/{prefix}-OPT-D.jpg",
                f"{question_id}/{prefix}-OPT-E.jpg",
                f"{question_id}/{prefix}-Question.png",
                f"{question_id}/{prefix}-OPT-A.png",
                f"{question_id}/{prefix}-OPT-B.png",
                f"{question_id}/{prefix}-OPT-C.png",
                f"{question_id}/{prefix}-OPT-D.png",
                f"{question_id}/{prefix}-OPT-E.png",
            ]

            storage_headers = {
                "Authorization": f"Bearer {key}",
                "apikey": key,
            }

            for path in candidate_paths:
                try:
                    delete_url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{quote(path, safe='/')}"
                    dresp = requests.delete(delete_url, headers=storage_headers, timeout=10)

                    if dresp.status_code in (200, 204):
                        storage_deleted.append(path)
                    else:
                        storage_errors.append({
                            "path": path,
                            "status": dresp.status_code,
                            "text": dresp.text
                        })
                except Exception as e:
                    storage_errors.append({
                        "path": path,
                        "error": str(e)
                    })

        delete_url = f"{supabase_url}/rest/v1/{table}?id=eq.{quote(str(actual_id))}"
        delete_headers = {
            "apikey": auth_key,
            "Authorization": f"Bearer {auth_key}",
            "Accept": "application/json",
            "Prefer": "return=representation",
        }

        delete_resp = requests.delete(delete_url, headers=delete_headers, timeout=10)

        if delete_resp.status_code not in (200, 204):
            return {
                "error": "Supabase delete failed",
                "status": delete_resp.status_code,
                "text": delete_resp.text,
                "actual_id": actual_id,
                "question_id": question_id
            }

        deleted_rows = []
        try:
            deleted_rows = delete_resp.json() if delete_resp.text else []
        except Exception:
            deleted_rows = []

        if delete_resp.status_code == 200 and isinstance(deleted_rows, list) and len(deleted_rows) == 0:
            return {
                "error": "Delete returned 200 but no rows were actually deleted",
                "deleted": False,
                "actual_id": actual_id,
                "question_id": question_id
            }

        return {
            "deleted": True,
            "actual_id": actual_id,
            "question_id": question_id,
            "storage_deleted": storage_deleted,
            "storage_errors": storage_errors
        }

    except Exception as e:
        return {"error": str(e)}
@app.post('/delete_object')
def delete_object(payload: dict):
    """Delete an object from Supabase storage.

    Accepts either `{ "public_url": "..." }` or `{ "bucket": "name", "path": "path/to/object.png" }`.
    Returns `{ deleted: True }` on success or an error detail on failure.
    """
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail='Supabase storage not configured on server')
        public_url = payload.get('public_url')
        bucket = payload.get('bucket')
        path = payload.get('path')
        if public_url and (not bucket or not path):
            # try to parse bucket/path from known Supabase public URL shape
            # e.g. https://<supabase>/storage/v1/object/public/{bucket}/{path}
            m = re.search(r'/storage/v1/object/(?:public/)?([^/]+)/(.*)$', public_url)
            if m:
                bucket = bucket or m.group(1)
                path = path or m.group(2)
        if not bucket or not path:
            raise HTTPException(status_code=400, detail='bucket and path (or public_url) required')

        delete_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{quote(path, safe='/')}"
        headers = {
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'apikey': SUPABASE_SERVICE_KEY,
        }
        resp = requests.delete(delete_url, headers=headers, timeout=15)
        if resp.status_code in (200, 204):
            return {'deleted': True}
        # return the status text for debugging
        raise HTTPException(status_code=502, detail=f'delete failed: {resp.status_code} {resp.text}')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/promote_object')
def promote_object(payload: dict):
    """Promote a temporary object (public_url or bucket+from_path) into a canonical path.

    Accepts one of:
      - { public_url, category, question_id, fieldName }
      - { bucket, from_path, to_path }
      - { public_url, to_path }

    The endpoint will fetch the source object, PUT it to the target path using
    the server's SUPABASE_SERVICE_KEY, delete the source object if it is in the
    same Supabase storage, and return the new public URL.
    """
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail='Supabase storage not configured on server')

        public_url = payload.get('public_url')
        bucket = payload.get('bucket')
        from_path = payload.get('from_path') or payload.get('path')
        to_path = payload.get('to_path')
        target_public_url = payload.get('target_public_url') or payload.get('replace_public_url')
        category = payload.get('category')
        question_id = payload.get('question_id') or payload.get('qid')
        fieldName = payload.get('fieldName') or payload.get('field')

        # If public_url provided, try to parse bucket/from_path
        if public_url and (not bucket or not from_path):
            m = re.search(r'/storage/v1/object/(?:public/)?([^/]+)/(.*)$', public_url)
            if m:
                bucket = bucket or m.group(1)
                # strip any query string or fragment from parsed path
                parsed_path = (m.group(2) or '')
                from_path = from_path or parsed_path.split('?', 1)[0].split('#', 1)[0]

        # If caller explicitly provided a target_public_url (existing canonical file), prefer it as destination
        if target_public_url and (not to_path):
            m = re.search(r'/storage/v1/object/(?:public/)?([^/]+)/(.*)$', target_public_url)
            if m:
                bucket = bucket or m.group(1)
                parsed_to = (m.group(2) or '')
                to_path = to_path or parsed_to.split('?',1)[0].split('#',1)[0]

        # If caller asked server to determine canonical to_path using category/question_id/fieldName
        if (not to_path) and (category or question_id) and fieldName:
            # reuse upload_dataurl mapping logic: simple folder map
            category_folder_map = {
                'Numerical Ability': 'NA-Test-01',
                'Numerical_Ability': 'NA-Test-01',
                'numerical_ability': 'NA-Test-01',
                'Numerical%20Ability': 'NA-Test-01',
                'Mechanical Ability': 'MA-Test-12',
                'Verbal Ability': 'VA-Test-01',
            }
            def norm(s):
                return str(s or '').lower().replace('%20',' ').replace('_',' ').strip()
            mapped_folder = None
            folder_key = category if category else (from_path.split('/',1)[0] if from_path else None)
            if folder_key:
                for k,v in category_folder_map.items():
                    if norm(k) == norm(folder_key): mapped_folder = v; break
            # fallback: infer from question_id prefix
            if not mapped_folder and question_id:
                for v in category_folder_map.values():
                    if str(v).upper().startswith(str(question_id).upper().split('-')[0]): mapped_folder = v; break

            # determine extension from source path or default to .png
            ext = ''
            try:
                if from_path and '.' in from_path:
                    ext = '.' + from_path.rsplit('.',1)[1]
                else:
                    # attempt to HEAD the public_url for content-type
                    if public_url:
                        r = requests.head(public_url, timeout=10)
                        ctype = r.headers.get('Content-Type') if r is not None else None
                        if ctype and '/' in ctype:
                            if ctype.endswith('jpeg') or ctype.endswith('jpg'): ext = '.jpg'
                            elif ctype.endswith('png'): ext = '.png'
                            elif ctype.endswith('webp'): ext = '.webp'
                            elif ctype.endswith('gif'): ext = '.gif'
            except Exception:
                ext = ext or ''

            is_question = 'question' in fieldName.lower()

            # Prefer prefix-based canonical filenames when we have a mapped folder
            # (e.g. NA-Test-01 -> prefix 'NA'). This ensures option images for
            # mapped categories are consistently named like `NA-OPT-A`..`NA-OPT-E`.
            if question_id and not mapped_folder:
                # Only use per-question filenames when there is NO mapped folder.
                safe_qid = str(question_id).strip()
                if is_question:
                    filename = f"{safe_qid}_Question{ext}"
                else:
                    # derive letter if fieldName like opt_a
                    m = re.search(r'([A-Ea-e])', fieldName)
                    letter = (m.group(1).upper() if m else 'A')
                    filename = f"{safe_qid}_OPT_{letter}{ext}"
            else:
                # Prefer prefix-based naming. If we have a mapped_folder use its
                # leading segment as the prefix (e.g. 'NA' from 'NA-Test-01');
                # otherwise fall back to a safe prefix derived from question_id
                # or a timestamp.
                prefix = None
                if mapped_folder:
                    prefix = str(mapped_folder).split('-')[0].upper()
                elif question_id:
                    prefix = str(question_id).split('-')[0].upper()
                else:
                    prefix = f"promoted_{int(time.time())}"

                if is_question:
                    filename = f"{prefix}-Question{ext}"
                else:
                    m = re.search(r'([A-Ea-e])', fieldName)
                    letter = (m.group(1).upper() if m else 'A')
                    filename = f"{prefix}-OPT-{letter}{ext}"

            if mapped_folder:
                to_path = f"{mapped_folder}/{filename}"
            else:
                to_path = f"uploads/{filename}"

        if not bucket or not from_path or not to_path:
            raise HTTPException(status_code=400, detail='public_url (or bucket+from_path) and to_path (or category+question_id+fieldName) required')

        debug_steps = []
        # Fetch the source bytes
        src_url = public_url or f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{quote(from_path, safe='/')}"
        debug_steps.append(f"fetching source_url={src_url}")
        try:
            r = requests.get(src_url, timeout=20)
            debug_steps.append(f"fetched source status={getattr(r,'status_code',None)}")
        except Exception as _e:
            debug_steps.append(f"fetch-exception: {_e}")
            raise HTTPException(status_code=502, detail=f'Failed to fetch source object: {_e}')
        if r.status_code not in (200, 206):
            debug_steps.append(f"fetch failed status={r.status_code} body={getattr(r,'text',None)}")
            raise HTTPException(status_code=502, detail=f'Failed to fetch source object: {r.status_code}')
        raw = r.content
        ctype = r.headers.get('Content-Type', 'application/octet-stream')

        # Upload to target path (PUT with upsert semantics). Use upload-first
        # to ensure the canonical path is present if the upload succeeds, then
        # remove the temp source. Retry a few times on transient errors.
        upload_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{to_path}"
        params = {'upsert': 'true', 'cacheControl': '3600'}
        headers = {
            'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
            'apikey': SUPABASE_SERVICE_KEY,
            'Content-Type': ctype,
        }
        upload_success = False
        upload_exc = None
        for attempt in range(1, 4):
            try:
                resp = requests.put(upload_url, params=params, headers=headers, data=raw, timeout=30)
                debug_steps.append(f"upload attempt={attempt} status={getattr(resp,'status_code',None)}")
                if resp.status_code in (200, 201):
                    upload_success = True
                    break
                upload_exc = f"status={resp.status_code} body={getattr(resp,'text',None)}"
            except Exception as _e:
                upload_exc = str(_e)
                debug_steps.append(f"upload attempt={attempt} exception={upload_exc}")
            time.sleep(0.5 * attempt)

        if not upload_success:
            debug_steps.append(f"upload failed after attempts: {upload_exc}")
            raise HTTPException(status_code=502, detail=f'Promote upload failed: {upload_exc}')

        # Build canonical public URL
        canonical = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{quote(to_path, safe='/')}"

        # Attempt to delete the source if requested and it's inside Supabase storage and different from target
        try:
            delete_source = bool(payload.get('delete_source', True))
            if delete_source:
                del_bucket = bucket
                del_path = (from_path or '').strip()

                # fallback parse only if from_path was not given
                if not del_path:
                    m2 = re.search(r'/storage/v1/object/(?:public/)?([^/]+)/(.*)$', src_url)
                    if m2:
                        del_bucket = m2.group(1)
                        del_path = (m2.group(2) or '').split('?', 1)[0].split('#', 1)[0]

                if del_bucket and del_path and not (del_bucket == bucket and del_path == to_path):
                    delete_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{del_bucket}/{quote(del_path, safe='/')}"
                    headers_del = {
                        'Authorization': f'Bearer {SUPABASE_SERVICE_KEY}',
                        'apikey': SUPABASE_SERVICE_KEY,
                    }

                    delete_success = False
                    delete_error = None

                    for attempt in range(1, 4):
                        try:
                            dresp = requests.delete(delete_url, headers=headers_del, timeout=15)
                            debug_steps.append(
                                f"delete_source attempt={attempt} status={getattr(dresp,'status_code',None)} path={del_path}"
                            )

                            if dresp.status_code in (200, 204):
                                delete_success = True
                                break

                            delete_error = f"status={dresp.status_code} body={getattr(dresp, 'text', None)}"
                        except Exception as _e:
                            delete_error = str(_e)
                            debug_steps.append(f"delete_source attempt={attempt} exception={_e}")

                        time.sleep(0.2 * attempt)

                    if not delete_success:
                        debug_steps.append(f"delete_source failed after attempts: {delete_error}")
        except Exception as e:
            debug_steps.append(f"delete_source outer exception: {e}")

        result = {'public_url': canonical, 'promoted': True}
        # If caller requested debug information, include the collected steps
        if payload.get('debug'):
            result['_debug_steps'] = debug_steps
        # Also print debug to server logs for troubleshooting
        try:
            for s in debug_steps:
                print('promote_object:', s)
        except Exception:
            pass
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users")
def list_users(limit: int = 100):
    """Return a list of rows from `users_profile`. Use sparingly; paginated via `limit` param."""
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

        url = f"{supabase_url}/rest/v1/users_profile?select=*&limit={int(limit)}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return {"error": "Supabase error", "status": resp.status_code, "detail": resp.text}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@app.patch("/users/{user_id}")
def update_user(user_id: str, payload: dict):
    """Update a users_profile row by id."""
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        url = f"{supabase_url}/rest/v1/users_profile?id=eq.{quote(user_id)}"
        # ensure an update timestamp is recorded (column `update_at`)
        payload_to_send = dict(payload or {})
        try:
            payload_to_send['update_at'] = datetime.utcnow().isoformat() + 'Z'
        except Exception:
            pass
        resp = requests.patch(url, json=payload_to_send, headers=headers, timeout=10)
        if resp.status_code != 200:
            return {"error": "Supabase error updating user", "status": resp.status_code, "detail": resp.text}
        data = resp.json()
        if isinstance(data, list) and data:
            user = data[0]
        else:
            user = data if isinstance(data, dict) else {}
        if 'email' in user and isinstance(user['email'], str):
            user['email'] = user['email'].strip()
        if 'password' in user:
            user.pop('password', None)
        return user
    except Exception as e:
        return {"error": str(e)}


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    """Delete a users_profile row by id."""
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

        url = f"{supabase_url}/rest/v1/users_profile?id=eq.{quote(user_id)}"
        resp = requests.delete(url, headers=headers, timeout=10)
        if resp.status_code in (200, 204):
            return {"deleted": True}
        return {"error": "Supabase error deleting user", "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"error": str(e)}


@app.get("/submissions")
def list_submissions(limit: int = 200):
    """Return rows from `student_submission` table (paginated via `limit`)."""
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

        url = f"{supabase_url}/rest/v1/student_submission?select=*&limit={int(limit)}&order=submitted_at.desc"
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return {"error": "Supabase error", "status": resp.status_code, "detail": resp.text}
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


class AuthRequest(BaseModel):
    email: str
    password: str


@app.post("/auth")
def auth_user(creds: AuthRequest):
    """Authenticate a user by email and password against users_profile table.
    NOTE: This assumes `users_profile` stores passwords in plaintext or comparable form.
    For production use proper password hashing or Supabase Auth.
    """
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Query the users_profile table for matching email AND password
        # Use eq filters; ensure values are URL-encoded
        email_q = quote(creds.email)
        pwd_q = quote(creds.password)
        url = f"{supabase_url}/rest/v1/users_profile?email=eq.{email_q}&password=eq.{pwd_q}&select=*&limit=1"
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return {"error": "Supabase error", "status": resp.status_code, "detail": resp.text}
        data = resp.json()
        if data:
            user = data[0]
            if 'email' in user and isinstance(user['email'], str):
                user['email'] = user['email'].strip()
            if 'password' in user:
                user.pop('password', None)
            return {"user": user}

        # Fallback: if exact match failed, try a case-insensitive email match (ilike)
        # to account for stray whitespace/newlines in stored emails while still matching password.
        try:
            ilike_email = quote(f"%{creds.email}%")
            alt_url = f"{supabase_url}/rest/v1/users_profile?email=ilike.{ilike_email}&password=eq.{pwd_q}&select=*&limit=1"
            resp2 = requests.get(alt_url, headers=headers, timeout=8)
            if resp2.status_code == 200:
                data2 = resp2.json()
                if data2:
                    user = data2[0]
                    if 'email' in user and isinstance(user['email'], str):
                        user['email'] = user['email'].strip()
                    if 'password' in user:
                        user.pop('password', None)
                    return {"user": user}
        except Exception:
            pass

        return {"error": "Invalid credentials"}
    except Exception as e:
        return {"error": str(e)}


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    usertype: str = 'user'


@app.post("/signup")
def signup_user(req: SignupRequest):
    """Create a new users_profile row.
    Returns the created row (without password) on success.
    """
    try:
        supabase_url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_SERVICE_KEY or os.getenv("SUPABASE_SERVICE_KEY")
        if not supabase_url or not key:
            return {"error": "Supabase URL or service key not configured on server"}

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        payload = {
            "name": req.name.strip(),
            "email": req.email.strip(),
            "password": req.password,
            "usertype": req.usertype,
        }

        url = f"{supabase_url}/rest/v1/users_profile"
        resp = requests.post(url, json=payload, headers=headers, timeout=8)
        if resp.status_code not in (200, 201):
            return {"error": "Supabase error creating user", "status": resp.status_code, "detail": resp.text}
        data = resp.json()
        if isinstance(data, list) and data:
            user = data[0]
        else:
            user = data if isinstance(data, dict) else {}
        # strip returned email and remove password
        if 'email' in user and isinstance(user['email'], str):
            user['email'] = user['email'].strip()
        if 'password' in user:
            user.pop('password', None)
        return user
    except Exception as e:
        return {"error": str(e)}

@app.get("/group-careers/{group_code}")
def get_group_careers(group_code: str):
    code = (group_code or "").strip().upper()
    return {
        "group": code,
        "careers": GROUP_CAREERS.get(code, [])
    }
