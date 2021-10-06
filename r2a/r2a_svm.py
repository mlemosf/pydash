from r2a.ir2a import IR2A
from player.parser import *
import time
from statistics import mean
from sklearn import svm
from pandas import DataFrame
import numpy
import pickle

class R2A_SVM(IR2A):

    def __init__(self, id):
        IR2A.__init__(self, id)
        self.model = 0
        self.throughputs = []
        self.times = []
        self.request_time = 0
        self.jitter = [0]
        self.qi = []

    def handle_xml_request(self, msg):
        self.request_time = time.perf_counter()
        self.send_down(msg)

    def handle_xml_response(self, msg):

        parsed_mpd = parse_mpd(msg.get_payload())
        self.qi = parsed_mpd.get_qi()

        t = time.perf_counter() - self.request_time
        self.throughputs.append(2*msg.get_bit_length() / t)
        self.times.append(t)
        if len(self.times) > 0:
            self.jitter.append(abs(t-self.times[-1]))
        
        self.send_up(msg)

    def handle_segment_size_request(self, msg):
        self.request_time = time.perf_counter()
        avg = ema(self.throughputs, 10, 0.5)
        print("Exponential Average chosen is", avg)

        X = numpy.array([[msg.get_bit_length(), self.request_time]])
        prediction = self.model.predict(X)[0]
        
        selected_qi = self.qi[prediction]
        for i in self.qi:
            if avg > i:
                selected_qi = i

        msg.add_quality_id(selected_qi)
        self.send_down(msg)

    def handle_segment_size_response(self, msg):
        t = time.perf_counter() - self.request_time
        self.throughputs.append(msg.get_bit_length() / t)
        self.send_up(msg)

    def initialize(self):
        import logging
        self.model,_,_= pickle.load(open('r2a/svm.pkl', 'rb'))
        
        
    def finalization(self):
        pass


# Exponential moving average over a set of data
def ema(val: list, days: int, alpha: int) -> int:
    data = DataFrame(val[-days:])
    return list(DataFrame.ewm(data, alpha=alpha).mean()[0])[-1]
