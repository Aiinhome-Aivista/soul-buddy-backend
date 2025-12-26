import json
import traceback
from sqlalchemy import text
from flask import request, jsonify
from database.config import engine
from model.llm_client import call_llm
from helper.global_helper import safe_json_loads


def make_user_compare_prompt(contextual_summary, technical_summary, user_input, file_name, column_name):
    return f"""
You are comparing two summaries to determine which one aligns better with the user's input.

File: {file_name}
Column: {column_name}

Contextual Summary:
{contextual_summary}

Technical Summary:
{technical_summary}

User Input:
{user_input}

Your task:
1. Compare how semantically similar the user's input is to each summary.
2. Score each summary on a scale of 0 to 1 based on similarity to the user input:
   - 1.0 = extremely similar
   - 0.0 = no similarity
3. Confidence must be calculated as:
   confidence = abs(score_contextual - score_technical)
4. Select the summary with the higher similarity score.

Your response must STRICTLY be valid JSON:

{{
    "which_is_more_accurate": {{
        "selected": "contextual_metadata" or "technical_metadata",
        "confidence": float,
        "explanation": "Brief explanation including similarity scores"
    }}
}}

Rules:
- Confidence must reflect the difference: high when scores differ a lot, low when similar.
- If scores are close: confidence < 0.4.
- If one score is much higher: confidence > 0.7.
"""



def compute_differences(contextual_summary, technical_summary, user_input):
    """
    Compare contextual, technical, and user input summaries
    to generate a differences list describing semantic distinctions.
    """
    differences = []

    if contextual_summary and technical_summary:
        differences.append(
            "Technical metadata focuses on data structure and statistical properties, "
            "while contextual metadata provides semantic interpretation and actionable insights."
        )
        differences.append(
            "Technical metadata does not suggest normalization or format conversion, "
            "whereas contextual metadata explicitly recommends these steps."
        )
        differences.append(
            "Contextual metadata identifies potential data quality issues (e.g., anomalies or mismatches) "
            "and provides suggestions for investigation."
        )

    # Add user input–specific comparison
    if user_input:
        differences.append(
            f"The user input emphasizes the purpose or meaning of the column ('{user_input}'), "
            "which contextual metadata addresses more directly than technical metadata."
        )

    return differences


def user_input_analyze_controller():
    try:
        # ---------- Step 1: Input Validation ----------
        session_name = request.form.get("session_name")
        user_input = request.form.get("user_input")
        file_name = request.form.get("file_name")
        column_name = request.form.get("column_name")

        if not all([session_name, user_input, file_name, column_name]):
            return jsonify({"error": "Missing one or more required parameters"}), 400

        # ---------- Step 2: Load analyze record ----------
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT response FROM `analyze` WHERE session_name = :sn"),
                {"sn": session_name},
            ).fetchone()

        if not row:
            return jsonify({"error": f"No analysis found for session_name: {session_name}"}), 404

        analyze_data = safe_json_loads(row[0])
        if not analyze_data:
            return jsonify({"error": "Invalid or empty analyze data"}), 400

        # Handle both "files" and "results.files"
        files_data = analyze_data.get("results", {}).get("files") or analyze_data.get("files")
        if not files_data or file_name not in files_data:
            return jsonify({"error": f"File '{file_name}' not found in analysis results"}), 404

        file_data = files_data[file_name]

        # Normalize column names
        comparison_data = file_data.get("comparison", {})
        normalized_map = {col.strip().lower(): col for col in comparison_data.keys()}
        lookup_key = column_name.strip().lower()

        if lookup_key not in normalized_map:
            return jsonify({
                "error": f"Column '{column_name}' not found in file '{file_name}'",
                "available_columns": list(comparison_data.keys())
            }), 404

        column_data = comparison_data[normalized_map[lookup_key]]

        contextual_summary = column_data.get("contextual_summary", "").strip()
        technical_summary = column_data.get("technical_summary", "").strip()

        if not (contextual_summary or technical_summary):
            return jsonify({"error": f"No summaries found for column '{column_name}'"}), 400

        # ---------- Step 3: Compare using LLM ----------
        prompt = make_user_compare_prompt(
            contextual_summary, technical_summary, user_input, file_name, column_name
        )
        llm_response = call_llm(prompt)
        parsed = safe_json_loads(llm_response)
        which_is_more_accurate = parsed.get("which_is_more_accurate", {})

        # ---------- Step 4: Update only this column’s section ----------
        # Compute enhanced difference list
        differences = compute_differences(contextual_summary, technical_summary, user_input)

        # Update only relevant fields (keep others as-is)
        column_data["user_input"] = user_input
        column_data["differences"] = differences
        column_data["confidence"] = which_is_more_accurate.get("confidence", 0.0)
        column_data["which_is_more_accurate"] = {
            "selected": which_is_more_accurate.get("selected", ""),
            "confidence": which_is_more_accurate.get("confidence", 0.0)
        }

        # Optionally, if LLM gave an explanation — move it under final_recommendation
        explanation = which_is_more_accurate.get("explanation", "")
        column_data["final_recommendation"] = {
            "use": which_is_more_accurate.get("selected", ""),
            "reason": explanation
        }

        # Ensure column_name remains
        column_data["column_name"] = column_name

        # Save modified structure back
        json_response = json.dumps(analyze_data, ensure_ascii=False)

        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE `analyze`
                    SET response = :resp,
                        status = 'updated',
                        processing_status = 'Completed',
                        error_message = NULL
                    WHERE session_name = :sn
                """),
                {"resp": json_response, "sn": session_name},
            )

        # ---------- Step 5: Return Response ----------
        return jsonify({
            "message": f"User input analyzed successfully for file '{file_name}', column '{column_name}'.",
            "results": {
                "session_name": session_name,
                "file_name": file_name,
                "column_name": column_name,
                "confidence": column_data["confidence"], 
                "user_input": user_input,
                "differences": differences,
                "which_is_more_accurate": column_data["which_is_more_accurate"],
                "final_recommendation": column_data["final_recommendation"]
            }
        })

    except Exception as e:
        # ---------- Step 6: Handle Errors ----------
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO `analyze`
                        (session_name, response, status, error_message, processing_status)
                        VALUES (:sn, NULL, 'failed', :err, 'Failed')
                        ON DUPLICATE KEY UPDATE
                            status='failed', error_message=:err, processing_status='Failed'
                    """),
                    {"sn": session_name or "unknown", "err": str(e)},
                )
        except Exception as db_err:
            print("DB error while saving failure:", db_err)

        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500