import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_subscription_email(
    to_email,
    full_name,
    plan_name,
    amount,
    currency,
    start_date,
    end_date,
    payment_method,
    transaction_id
):
    try:
        sender_email = "saikatofficial1998@gmail.com"
        sender_password = "gkzlglukauqwflnd"

        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = "🎉 Subscription Activated Successfully"

        text_body = f"""
Hello {full_name},

Your subscription is active.

Plan: {plan_name}
Amount: {currency} {amount}
Payment: {payment_method}
Transaction ID: {transaction_id}

Valid From: {start_date}
Valid Till: {end_date}
"""

        html_body = f"""
<html>
<body>
<h2>Payment Successful 🎉</h2>
<p><b>Plan:</b> {plan_name}</p>
<p><b>Amount:</b> {currency} {amount}</p>
<p><b>Payment Method:</b> {payment_method}</p>
<p><b>Transaction ID:</b> {transaction_id}</p>
<p><b>Valid From:</b> {start_date}</p>
<p><b>Valid Till:</b> {end_date}</p>
</body>
</html>
"""

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        return True

    except Exception as e:
        print("Email Error:", e)
        return False
