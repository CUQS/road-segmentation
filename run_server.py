import socket
import numpy as np
import cv2

buf_size = 623*188*3

def recv_size(sokt, size):
    buf = b""
    while size:
        newbuf = sokt.recv(size)
        if not newbuf:
            return None
        buf += newbuf
        size -= len(newbuf)
    return buf

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('192.168.1.134', 4097))
server.listen()

print("wait connect...")
conn, addr = server.accept()
print("connected, ", addr)
print("wait info...")
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
count = 60
while count:
    stringData = recv_size(conn, buf_size)
    # data convert
    data = np.frombuffer(stringData, np.uint8)
    data = np.reshape(data, (188,623,3))
    cv2.imshow("img", data)
    cv2.waitKey(10)
    count -= 1

conn.close()
server.close()
print("closed")