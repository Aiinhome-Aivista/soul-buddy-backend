from flask import jsonify
import mysql.connector
from database.db_connection import get_db_connection   

def get_subscription_plans():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                id,
                plan_code,
                plan_name,
                original_price,
                offer_price,
                discount_percent,
                minutes_per_day,
                validity_days,
                is_trial
            FROM subscription_plans
            WHERE is_active = 1
            ORDER BY id
        """

        cursor.execute(query)
        plans = cursor.fetchall()

        # Format response
        response = []
        for plan in plans:
            response.append({
                "id": plan["id"],
                "title": plan["plan_code"],
                "planName": plan["plan_name"],
                "originalPrice": plan["original_price"],
                "finalPrice": plan["offer_price"],
                "discount": f'{plan["discount_percent"]}%',
                "usage": "Unlimited"
                         if plan["minutes_per_day"] is None
                         else f'{plan["minutes_per_day"]} minutes/day',
                "validityDays": plan["validity_days"],
                "isTrial": bool(plan["is_trial"])
            })

        return jsonify({
            "status": "success",
            "statusCode": 200,
            "data": response
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "statusCode": 500,
            "message": str(e)
        })