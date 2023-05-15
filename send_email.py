import smtplib, ssl
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default='DONE!', help='message to send')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "maxkim139@@gmail.com"
    password = "fbcfecbthtlxbhdg"
    # pickle.dump(password)

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        # TODO: Send email here
        server.sendmail(sender_email, 'maxkim139@gmail.com', args.message)

    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()