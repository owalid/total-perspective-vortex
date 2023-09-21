import socket
import time
import json
import glob
import os
import mne
import base64
from mne.io import concatenate_raws, read_raw_edf
from uuid import uuid4
from threading import Thread
from utils import receive_message, send_message


EXPERIMENTS = {
    'hands_vs_feet': [3, 7, 11],
    'left_vs_right': [5, 9, 13],
    'imagery_left_vs_right': [4, 8, 12],
    'imagery_hands_vs_feet': [6, 10, 14],
}

SUBJECT_AVAILABLES = range(1, 110)
SUBJECT_AVAILABLES = list(SUBJECT_AVAILABLES)
SUBJECT_AVAILABLES.remove(88)
SUBJECT_AVAILABLES.remove(92)
SUBJECT_AVAILABLES.remove(100)

SIZE_OF_RECEIVE = 128
RECEIVE_TIMOUT = 1800
HOST = '0.0.0.0'
PORT = 5000

class SocketServer:
    __instance = None

    @staticmethod
    def get_instance():
        '''
        Static access method. used to make singleton.
        '''
        if SocketServer.__instance == None:
            SocketServer()
        return SocketServer.__instance

    def __init__(self, directory_dataset):
        if SocketServer.__instance != None:
            return SocketServer.__instance
        else:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(socket.SOMAXCONN)
            
            if not os.path.exists(directory_dataset):
                raise ValueError(f'Directory dataset not exists: {directory_dataset}')
            if not os.path.isdir(directory_dataset):
                raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
            
            self.directory_dataset = directory_dataset
            if self.directory_dataset[-1] == '/':
                self.directory_dataset = self.directory_dataset[:-1]

            SocketServer.__instance = self

    def exit_client(self, conn):
        if conn:
            conn.close()
    
    def handle_requests(self, conn):
        while True:
            r = receive_message(conn, SIZE_OF_RECEIVE)
            if not r:
                break

            laddr = conn.getpeername()
            print(f'Client {laddr} sent: {r}')
            input_client = r.decode().strip()
            input_client = input_client.split(':')
            subject_index, experiment = input_client
            print(f'Client {laddr} requested subject {subject_index} for experiment {experiment}')
            if experiment not in EXPERIMENTS.keys():
                self.exit_client(conn)
                return
            
            if not subject_index.isdigit():
                self.exit_client(conn)
                return
            
            subject_index = int(subject_index)
            print(f'Client {laddr} requested subject {subject_index} for experiment {experiment}')
            if subject_index in SUBJECT_AVAILABLES:
                subject_index = f'S{subject_index:03d}'
                print(f"Client {laddr} requested subject {subject_index} for experiment {experiment}")
                files = glob.glob(f'{self.directory_dataset}/{subject_index}/*.edf')
                raws = []
                for i in EXPERIMENTS[experiment]:
                    current_file = files[i-1]
                    r = read_raw_edf(current_file, preload=True, stim_channel='auto', verbose=False)
                    raws.append(r)

                raw = concatenate_raws(raws)

                event_id = {'T1': 1, 'T2': 2}
                events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

                tmin = -0.5
                tmax = 2
                picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
                epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=False)

                epochs_data = epochs.get_data()
                epochs_label = epochs.events[:, -1] - 1
                sfreq = raw.info['sfreq']

                for ii in range(len(epochs_label)):
                    current_data = epochs_data[ii]
                    current_label = epochs_label[ii]
                    to_send = {
                        'epoch': current_data.tolist(),
                        'label': current_label.tolist(),
                        'sfreq': sfreq
                    }
                    to_send = json.dumps(to_send)
                    base64_bytes = base64.b64encode(to_send.encode('ascii')) # encode as base64
        
                    send_message(conn, base64_bytes + b'\x00\x00\x00\x00')
                    print(f"[+] epoch {ii+1} sent to client {laddr}")
                    time.sleep(2.5)
                    is_ready = False
                    while not is_ready:
                        input_client = receive_message(conn, SIZE_OF_RECEIVE)
                        if not input_client:
                            continue
                        if input_client.decode().strip() == 'next':
                            is_ready = True
                
                send_message(conn, b'end')
                print(f"[+] All epochs sent to client {laddr}")
                self.exit_client(conn)
                    

    def start_server(self):
        try:
            print("[+] Server started")
            print(f"[+] Listening for connections on 0.0.0.0:{PORT}")
            print("[+] Waiting for client request..")
            while True:
                conn, _ = self.server_socket.accept()
                print(f'New connection: {conn}')
                # start thread for client
                thread = Thread(target=self.handle_requests, args=(conn,))
                thread.start()

        except Exception as e:
            print(e)
            self.exit_client(conn)