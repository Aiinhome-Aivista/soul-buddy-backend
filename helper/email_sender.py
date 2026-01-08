import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_otp_email(to_email, otp):
    try:
        sender_email = "saikatofficial1998@gmail.com"
        sender_password = "gkzlglukauqwflnd" 

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = "Your Password Reset OTP"

        body = f"""
        Your OTP for password reset is: {otp}

        This OTP will expire in 10 minutes.
        """
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        return True
    except Exception as e:
        print("Email error:", e)
        return False
