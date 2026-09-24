from email.mime.text import MIMEText
import smtplib
import socket
import os

from app.config import SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL

def send_email(recipient_email: str, subject: str, body: str):
    """Sends an email notification."""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        print("Email configuration missing. Skipping email notification.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {recipient_email} for subject: {subject}")
    except socket.gaierror as e:
        print(f"Failed to send email. DNS resolution error for SMTP server '{SMTP_SERVER}'. Check your EMAIL_HOST environment variable. Details: {e}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")
