import smtplib, ssl
import argparse
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sender_email', type=str, default="ssohot1@gmail.com")
    parser.add_argument('--receiver_email', type=str, default="ssohot1@gmail.com")
    parser.add_argument('--password', type=str, default="nfiunmyiwrsqupld")
    parser.add_argument('--message', type=str, default='DONE!', help='message to send')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    smtp_server = "smtp.gmail.com"
    port = 587 # for starttls
    context = ssl.create_default_context() # create a secure SSL context

    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo() # can be omitted
        server.starttls(context=context) # secure the connection
        server.ehlo() # can be omitted
        server.login(args.sender_email, args.password)
        server.sendmail(args.sender_email, args.receiver_email, args.message)
    except Exception as e:
        print(e)
    finally:
        server.quit()