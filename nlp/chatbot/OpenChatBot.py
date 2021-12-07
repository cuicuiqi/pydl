from chatterbot import ChatBot
import socket
import urllib.parse
from threading import Thread
import logging
import traceback

def handle_request(conn, chatbot):
    response_start_line = "HTTP/1.1 200 OK\r\n"
    response_headers = "Server: NBAPI python server\r\n\r\n"

    try:
        request = conn.recv(1024)
        logging.info('received:' + str(request.decode()))
        path = request.decode().split(' ')[1]
        if (path == '/'):
            conn.send(bytes(response_start_line + response_headers, "utf-8"))
            return

        data = path.split('/')[2]
        data = urllib.parse.unquote(data)
        response = chatbot.get_response(data)
        response_body = response.text + ',' + str(response.confidence)
        response = response_start_line + response_headers + response_body
        conn.send(bytes(response, "utf-8"))
    except Exception as e:
        logging.error("handler thread:" + str(e))
        logging.error(traceback.format_exc())
        response_body = str(e)
        response = response_start_line + response_headers + response_body
        conn.send(bytes(response, "utf-8"))
    finally:
        conn.close()
        logging.info('response done.')


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('',8000))
sock.listen(128)
chatbot = ChatBot('OpenChatBot')

while True:
    try:
        conn, addr = sock.accept()
        logging.info('connected:' + str(addr))
        handle_client_process = Thread(target=handle_request,
                                       args=(conn, chatbot))
        handle_client_process.start()
    except Exception as e:
        logging.error("main thread:" + str(e))
        logging.error(traceback.format_exc())

