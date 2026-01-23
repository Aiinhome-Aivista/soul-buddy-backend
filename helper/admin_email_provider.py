import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from database.config import NOREPLY_EMAIL, NOREPLY_PASSWORD, ADMIN_EMAIL

def send_admin_payment_email(user_email, amount, payment_id, plan_name):
    try:
        msg = MIMEMultipart()
        msg["From"] = NOREPLY_EMAIL
        msg["To"] = ADMIN_EMAIL
        msg["Subject"] = "💰 New Payment Received - SoulJunction"

        body = f"""
New payment received successfully.

User Email: {user_email}
Plan: {plan_name}
Amount: INR {amount}
Transaction ID: {payment_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Login to admin panel for details.
        """

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("mail.souljunction.life", 465) as server:
            server.login(NOREPLY_EMAIL, NOREPLY_PASSWORD)
            server.send_message(msg)

        return True

    except Exception as e:
        print("Admin Email Error:", e)
        return False
