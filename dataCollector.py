import serial
import time
import csv

serialPort = serial.Serial(
    port="COM7", baudrate=115200, parity=serial.PARITY_ODD, bytesize=serial.SEVENBITS, timeout=0.1, stopbits=serial.STOPBITS_TWO
)

features = ["time", "ch1", "ch2", "ch3", "ch4", "label"]
with open("veriseti5.csv", "w", encoding="utf-8", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(features) 
    file.close()

def wait():
    print("Start in 5 seconds..")
    print("First label will be rest.")
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

def label():
    global pre_label_shown
    labels = ["rahat", "yumruk", "rahat", "ön", "rahat", "arka"]  
    label_index = 0  
    start_time = time.time()  
    label_start_time = time.time() 
    label_duration = 3  
    countdown_start_offset = 2  
    pre_label_shown = False  

    print(f"Current label: {labels[label_index]}")

    with open("veriseti5.csv", "a", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)

        while True:
            amplitude = serialPort.readline().decode("utf-8").strip() 
            #print(amplitude)  # Debugging raw data
            
            elapsed_time = time.time() - start_time  
            label_elapsed_time = time.time() - label_start_time  
            remaining_time = label_duration - label_elapsed_time

            try:
                channels = [x.strip() for x in amplitude.split(",") if x.strip()]
                if len(channels) != 4:
                    raise ValueError("Incorrect channel data format")
                
                ch1, ch2, ch3, ch4 = map(float, channels)  
            except Exception as e:
                print(f"Error parsing data: {amplitude} - {e}")
                continue  
            
            if not pre_label_shown and remaining_time <= countdown_start_offset:
                next_label_index = (label_index + 1) % len(labels)
                print(f"Next label: {labels[next_label_index]} (in {countdown_start_offset} sec)")

                start_time2 = time.time() 
                total_countdown_duration = 2  
                interval = total_countdown_duration / 3 

                for i in range(3, 0, -1):
                    while time.time() - start_time2 < interval * (4 - i):  
                        pass  
                    print(i)

                """ for i in range(3, 0, -1):
                    print(i) """

                pre_label_shown = True

            if label_elapsed_time >= label_duration:
                label_index = (label_index + 1) % len(labels)  
                print(f"\nCurrent label: {labels[label_index]}")
                label_start_time = time.time()  
                pre_label_shown = False  

            if amplitude is not None:
                row = [elapsed_time, ch1, ch2, ch3, ch4, labels[label_index]]
                csv_writer.writerow(row)
                #print(f"Data saved: {row}")

if __name__ == "__main__":
    wait()
    label()
