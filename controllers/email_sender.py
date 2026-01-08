import smtplib
from email.mime.text import MIMEText

def send_otp_email(to_email, otp):
    # --- MOCK EMAIL FOR TESTING ---
    print(f"\n[EMAIL MOCK] ---------------------------")
    print(f"To: {to_email}")
    print(f"Subject: Password Reset OTP")
    print(f"Your OTP Code is: {otp}")
    print(f"---------------------------------------\n")
    return True