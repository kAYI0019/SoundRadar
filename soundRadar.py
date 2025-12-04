import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import random
import math
import numpy as np
import queue
import sounddevice as sd
import time

class TranslucentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, position=0):
        super(TranslucentWidget, self).__init__(parent)
        # make the window frameless
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.popup_fillColor = QtGui.QColor(240, 240, 240, 255)
        self.popup_penColor = QtGui.QColor(200, 200, 200, 255)
        self.position = position
        
        self.strength = 0.0
    def resizeEvent(self, event):
        s = self.size()
        popup_width = 300
        popup_height = 120
        ow = int(s.width() / 2 - popup_width / 2)
        oh = int(s.height() / 2 - popup_height / 2)
    def createPoly(self, p, r=20, n=20):
        polygon = QtGui.QPolygonF()
        polygon.append(QtCore.QPointF(self.width()/2, self.height()/2)) #center of circle
        for i in range(n):                                              # add the points of polygon
            t = -105 + p * 30 + (30. / n) * i
            x = r*math.cos(math.radians(t))
            y = r*math.sin(math.radians(t))
            polygon.append(QtCore.QPointF(self.width()/2 +x, self.height()/2 + y))  
        polygon.append(QtCore.QPointF(self.width()/2, self.height()/2)) #center of circle
        return polygon
    def paintEvent(self, event):
        # This method is drawing the contents of your window.
        # get current window size
        s = self.size()
        w, h = s.width(), s.height()
        cx, cy = w / 2, h / 2

        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)

        strength = getattr(self, "strength", 0.0)
        strength = max(0.0, min(1.0, float(strength)))

        # according to the strength, set the color and transparency
        # very low: light green, low: dark green, mid: yellow, high: orange/red
        if strength < 0.25:
            # very small sound: inside, light green, almost transparent
            r, g, b, alpha = 60, 200, 60, 40
        elif strength < 0.5:
            # small to medium: dark green
            r, g, b, alpha = 40, 255, 80, 90
        elif strength < 0.75:
            # middle: yellow
            r, g, b, alpha = 255, 220, 60, 150
        else:
            # very large sound: orange/red, opaque
            r, g, b, alpha = 255, 120, 40, 220

        # Apply opacity multiplier
        alpha = int(alpha * opacity_multiplier)
        alpha = max(0, min(255, alpha))

        color = QtGui.QColor(r, g, b, alpha)

        # according to the strength, set the pen width/radius
        pen_width = 2 + 10 * strength  # 2~12px
        qp.setPen(QtGui.QPen(color, pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        qp.setBrush(QtCore.Qt.NoBrush)

        # according to the screen size, set the radius
        # smaller sound is closer to the center, larger sound is much further from the center
        min_r = min(w, h) * 0.18  # closer to the center
        max_r = min(w, h) * 0.40  # a bit further from the center
        radius = (min_r + (max_r - min_r) * strength) * size_multiplier

        rect = QtCore.QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius)

        start_deg = -105 + self.position * 30 + 180
        span_deg = 30
        start_angle = int(start_deg * 16)
        span_angle = int(span_deg * 16)

        qp.drawArc(rect, start_angle, span_angle)
        qp.end()


class ParentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ParentWidget, self).__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)               
        self.popframes = {}
        self._popflag = False
        self.global_peak = 0.1

        for i in range(12):
            self.create_shape(position=i)
        for item in self.popframes:
            self.popframes[item]['shape'].move(0, 0)
            self.popframes[item]['shape'].resize(self.width(), self.height())
        self.setBackgroundcolor()
    def resizeEvent(self, event):
        if self._popflag:
            for item in self.popframes:
                self.popframes[item]['shape'].move(0, 0)
                self.popframes[item]['shape'].resize(self.width(), self.height())
    def create_shape(self, position=0):
        popframe = TranslucentWidget(self, position)
        popframe.move(0, 0)
        popframe.resize(self.width(), self.height())
        self._popflag = True
        self.popframes[position] = {}
        self.popframes[position]['shape'] = popframe
        self.popframes[position]['shape'].show()
        self.popframes[position]['tupdate'] = 0
        self.popframes[position]['fistFlag'] = False
    def updateBrush(self, color, position):
        # update the color of the selected zone
        # color is a (1,3) array containing the RGB colors
        # position is an int giving the clocklike position
        r = int(color[0])
        g = int(color[1])
        b = int(color[2])
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        try:
            raw = float(prevmax[position])
            raw = max(0.0, min(1.0, raw))
            # according to the strength mode, set the processing method
            from math import pow as mpow
            if STRENGTH_MODE == 1:
                # mode 1: only emphasize the strongest direction among all
                # track the recent global peak (slowly decreasing)
                self.global_peak = max(self.global_peak * 0.9, raw, 1e-3)
                ratio = raw / (self.global_peak + 1e-6)
                # only keep the values around the largest value (0.6 or higher) and almost 0 for the rest
                if ratio < 0.6:
                    strength = 0.0
                else:
                    strength = (ratio - 0.6) / 0.4  # 0.6→0, 1.0→1
            else:
                # mode 2: show multiple directions, but the strongest direction is much more prominent
                # square the values to make smaller values smaller and larger values stay the same
                strength = raw * raw
            strength = max(0.0, min(1.0, strength))
            self.popframes[position]['shape'].strength = strength
        except Exception:
            self.popframes[position]['shape'].strength = 0.0      
        self.update()
    def setBackgroundcolor(self):
        self.p = QtWidgets.QWidget.palette(self)
        self.p.setColor(self.backgroundRole(),QtGui.QColor(0,0,0,0))
        self.setPalette(self.p)





def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block.
    Shape of outdata is (frames, channels)"""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[:])

def getMaxSound(n_chans):
    """This is called by process to update values for each channel

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """
    maxVals = np.zeros(n_chans)
    while True:        
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
    if 'data' in locals():
        maxVals = np.nanmax(np.append([maxVals],data,axis=0),axis=0)
    return maxVals/maxSoundValue # return sound in percentage

def enhancer(x):
    if x < minThreshold:
        return 0
    elif x > minThreshold:
        return 1
    else:
        return (2*x-minThreshold)**(1/2)

def initfilter(x, t):
    x[x<t] = 0
    return np.fromiter((expfilter(xi) for xi in x), x.dtype)
 
def expfilter(x):
    return 1 - math.exp(-5*x)

# Exponential fade effect
def apply_fade(current_value, elapsed_time, decay_rate=2.0):
    """Apply exponential decay fade - natural fade out"""
    return current_value * math.exp(-decay_rate * elapsed_time)

def updateRadar(radarObject):
    while True:
        # process input
        time.sleep(refreshtime)
        maxValues = getMaxSound(n_channel)
        maxValues = initfilter(maxValues, minThreshold)
        if DEBUG:
            print(maxValues[[0,1,4,5,6,7]]*100) # this will generate a lot of output. Could be improved by updating the line instead of printing a new line :o)
        for pos in radarObject.popframes:
            # update each part of the "radar"
            if pos == 0:
                if ((maxValues[mapping['avg']] + maxValues[mapping['avd']]) / 2 ) > prevmax[pos]:
                    # valid for the center (diff < maxdifmain) and higher than previous max - if not valid : reduce prevmax
                    prevmax[pos] = ((maxValues[mapping['avg']] + maxValues[mapping['avd']]) / 2 )
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 1:
                if (maxValues[mapping['avd']] - maxValues[mapping['avg']]) / min(maxValues[mapping['avg']],maxValues[mapping['avd']]) > maxdifmain and (maxValues[mapping['avd']] - maxValues[mapping['avg']]) > prevmax[pos]:
                    # valid if avd larger than avg of at least maxdifmain percents
                    prevmax[pos] = (maxValues[mapping['avd']] - maxValues[mapping['avg']]) 
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 2:
                if maxValues[mapping['d']] > maxValues[mapping['avd']] and maxValues[mapping['d']] - maxValues[mapping['avd']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['d']] - maxValues[mapping['avd']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 3:
                if maxValues[mapping['d']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['d']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 4:
                if maxValues[mapping['d']] > maxValues[mapping['ard']] and maxValues[mapping['d']] - maxValues[mapping['ard']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['d']] - maxValues[mapping['ard']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 5:
                if (maxValues[mapping['ard']] - maxValues[mapping['arg']]) / min(maxValues[mapping['arg']],maxValues[mapping['ard']]) > maxdifmain and (maxValues[mapping['ard']] - maxValues[mapping['arg']]) > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['ard']] - maxValues[mapping['arg']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 6:
                if  ((maxValues[mapping['arg']] + maxValues[mapping['ard']]) / 2 ) > prevmax[pos]:
                    prevmax[pos] = (maxValues[mapping['arg']] + maxValues[mapping['ard']]) / 2
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 7:
                if (maxValues[mapping['arg']] - maxValues[mapping['ard']]) / min(maxValues[mapping['ard']],maxValues[mapping['arg']]) > maxdifmain and (maxValues[mapping['arg']] - maxValues[mapping['ard']]) > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['arg']] - maxValues[mapping['ard']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 8:
                if maxValues[mapping['g']] > maxValues[mapping['arg']] and maxValues[mapping['g']] - maxValues[mapping['arg']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['g']] - maxValues[mapping['arg']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 9:
                if maxValues[mapping['g']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['g']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 10:
                if maxValues[mapping['g']] > maxValues[mapping['avg']] and maxValues[mapping['g']] - maxValues[mapping['avg']] > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['g']] - maxValues[mapping['avg']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if pos == 11:
                if (maxValues[mapping['avg']] - maxValues[mapping['avd']]) / min(maxValues[mapping['avg']],maxValues[mapping['avd']]) > maxdifmain and (maxValues[mapping['avg']] - maxValues[mapping['avd']]) > prevmax[pos]:
                    prevmax[pos] = maxValues[mapping['avg']] - maxValues[mapping['avd']]
                    radarObject.popframes[pos]['tupdate'] = time.time()
                    radarObject.popframes[pos]['fistFlag'] = True
                    prevmax[pos] = enhancer(prevmax[pos])
                elif radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTFU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['fistFlag'] = False
                    radarObject.popframes[pos]['tupdate'] = time.time()
                elif not radarObject.popframes[pos]['fistFlag'] and time.time() - radarObject.popframes[pos]['tupdate'] > minTBU:
                    elapsed_time = time.time() - radarObject.popframes[pos]['tupdate']
                    prevmax[pos] = apply_fade(prevmax[pos], elapsed_time, fade_decay_rate)
                    radarObject.popframes[pos]['tupdate'] = time.time()
            if prevmax[pos] < 0.01:
                prevmax[pos]=0
            radarObject.updateBrush([0, prevmax[pos] * maxColorRange, 0], pos)
        if DEBUG:
            print(prevmax)
            print('----')
        QtWidgets.QApplication.processEvents() #update GUI in a loop process





# GLOBAL PARAMETERS (should be in capital... at least the title is in capital :o) )
n_chans=8 # number of channels on sound device
n_channel = n_chans # yup, that's badly coded :o)
maxSoundValue = 2. ** 32 /2 # to be updated according to the dtype stream recording

STRENGTH_MODE = 2
mapping = {}
mapping['avg'] = 1 - 1  # avg = front left
mapping['avd'] = 2 - 1  # avd = front right
mapping['d'] = 8 - 1    # d = right
mapping['g'] = 7 - 1    # g = left
mapping['arg'] = 5 - 1  # arg = back left
mapping['ard'] = 6 - 1  # ard = back right
minTFU = 0.5 # minimum Time needed for First Update (upper sound value)
minTBU = 0.1 # minimum Time needed Between Update (lower sound value)
maxdifmain = 0.01 # max percentage difference between main front/back channels
maxColorRange = 255 # max value for color
minThreshold = 0.005 # lowpass filter threshold on maxValues
prevmax = np.zeros(12) # initialize the "previous max" value
redfactor = 5 #reduction factor if no upper value recorded
refreshtime = 0.1 # time between two refresh

# Fade effect settings
fade_decay_rate = 2.0  # Exponential decay rate (higher = faster fade out)

# Visualization settings
size_multiplier = 2.0  # Radar size multiplier (0.5 ~ 2.0, default: 1.0)
opacity_multiplier = 2.0  # Opacity multiplier (0.0 ~ 1.0, default: 1.0)

DEBUG = False
def find_device_auto(search_keywords, device_type='input'):
    """Automatically find device by searching through keyword list"""
    devices = sd.query_devices()
    
    # Try each keyword
    for keyword in search_keywords:
        keyword_lower = keyword.lower()
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            max_input = device.get('max_input_channels', 0)
            
            if keyword_lower in device_name:
                if device_type == 'input' and max_input > 0:
                    return i, device
                elif device_type == 'any':
                    return i, device
    
    return None, None

if __name__ == "__main__":
    q = queue.Queue()
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = ParentWidget()
    mainwindow.resize(500, 500)
    mainwindow.show()
    
    # try to find the device automatically
    search_keywords = ['CABLE Output', 'VB-Audio Virtual Cable', 'VB-Audio']
    device_id, found_device = find_device_auto(search_keywords, 'input')
    
    if device_id is not None:
        print(f"✓ Device found automatically: {found_device['name']} (ID: {device_id})")
        device_info = found_device
    else:
        # Manual input if device not found automatically
        print(sd.query_devices()) # print all devices available
        device_id = int(input('device id:')) # if we want user to select device
        device_info = sd.query_devices(device_id, 'input') # retrieve device infos

    #device_id=38 # input device to process -> should be commented out if previous line is active :o)
    
    # Update channel count based on actual device
    n_chans = device_info['max_input_channels']
    n_channel = n_chans


    stream = sd.InputStream(dtype=np.int32, device=device_id, channels=device_info['max_input_channels'],samplerate=device_info['default_samplerate'], callback=audio_callback)


    with stream:
        updateRadar(mainwindow)



