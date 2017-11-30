import time
import sys
import util

bar = util.ProgressBar(total = 100)
for i in range(100):
    bar.move()
    bar.log("ABC")
    # bar.log('We have arrived at: ' + str(i + 1))
    time.sleep(0.1)
