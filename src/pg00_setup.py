import datetime
import os

start = datetime.datetime.now()
os.system('clear')

data = './data/'
img = './img/'
model = './model/'
doc = './doc/'
src = './src/'

################################################################################

for dir in [data,img,model,doc,src]:
    if not os.path.exists(dir):
        os.makedirs(dir)  
        print(f"The {dir} directory created with success!")
    else:
        print(f"The {dir} directory already exists.")

################################################################################

end = datetime.datetime.now()
time = end - start

hour = str(time.seconds // 3600).zfill(2)
min = str((time.seconds % 3600) // 60).zfill(2)
sec = str(time.seconds % 60).zfill(2)

msg_time = f'Time:{hour}:{min}:{sec} '
print(msg_time)

################################################################################