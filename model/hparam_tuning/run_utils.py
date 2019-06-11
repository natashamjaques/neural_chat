import smtplib


DEFAULT_EMAIL_LIST = []
SENDING_ADDRESS = ''
SENDING_PASSWORD = ''


def send_email(subject, text, to_addr_list=DEFAULT_EMAIL_LIST):
    body = "\r\n".join(['From: %s' % SENDING_ADDRESS,
                        'To: %s' % to_addr_list,
                        'Subject: %s' % subject,
                        '',
                        text])

    try:
        server = smtplib.SMTP('smtp.gmail.com:587')  # NOTE:  This is the GMAIL SSL port.
        server.ehlo() # this line was not required in a previous working version
        server.starttls()
        server.login(SENDING_ADDRESS, SENDING_PASSWORD)
        server.sendmail(SENDING_ADDRESS, to_addr_list, body)
        server.quit()
        print("Email sent successfully!")
        return True
    except Exception as e:
        print("Email failed to send!")
        print(str(e))
        return False


def get_secs_mins_hours_from_secs(total_secs):
    hours = total_secs / 60 / 60
    mins = (total_secs % 3600) / 60
    secs = (total_secs % 3600) % 60

    if hours < 1: hours = 0
    if mins < 1: mins = 0
    
    return hours, mins, secs
