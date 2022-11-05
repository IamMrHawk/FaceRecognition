# imports
import cv2
import os
from tqdm import tqdm
import csv
from pathlib import Path


class CreateDataset:
    def __init__(self):
        self.vids = []
        self.user_dict = []
        self.counter_user = 0
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.write = self.create_csv()
        self.list_videos()
        self.main()
        self.save_classes()

    # create CSV file for annotations
    @staticmethod
    def create_csv():
        csv_file = open('annotations.csv', 'a', newline='')
        fields = ['image_name', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        return writer

    def list_videos(self):
        # video names with extension 
        inp = 'VideoDataset'
        dir_list = os.listdir(inp)
        # video path with extension  
        for directory in dir_list:
            addr = os.path.join(inp, directory)
            self.vids.append(addr)

    # Saving images and annotations to dataset
    def save_dataset(self, save_path, image, box, uname, counter, cls):
        x, y, w, h = box
        crpimg = image[y:y + h, x:x + h]
        file_name = uname+str(counter)+'.jpg'
        cv2.imwrite(f'{save_path}/{file_name}', crpimg)
        self.write.writerow({'image_name': f'{file_name}', 'class': f'{cls}'})

    def main(self):
        # create output directory to save annotations
        Path('dataset').mkdir(parents=True, exist_ok=True)

        for path in tqdm(self.vids, desc='Dataset Classes'):
            # Capture object name
            name = path.split('\\')[1]
            name = name.split('.')[0]

            # create user dictionary
            self.user_dict.append(name)

            # join path for Train and Test

            train_path = os.path.join('dataset')
            test_path = os.path.join('dataset')

            # reading video file
            cap = cv2.VideoCapture(path)

            # setting counter to 0
            counter_img = 0

            # initializing counter for 1000 iterations
            while counter_img < 300:
                # reading frames from video
                _, img = cap.read()

                # capturing face locations/ coordinates
                face = self.face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
                bb = face[0][0], face[0][1], face[0][2], face[0][3]

                # condition to create dataset for Training and Validation
                if counter_img < 230:
                    self.save_dataset(train_path, img, bb, name, counter_img, self.counter_user)

                else:
                    self.save_dataset(test_path, img, bb, name, counter_img, self.counter_user)

                # counter to save image / annotations by sequence
                counter_img += 1

                # wait key and termination
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # counter for user/classes
            self.counter_user += 1

            # destroy all open sources
            cap.release()
            cv2.destroyAllWindows()

        # user list along with their classes
        print(self.user_dict)

    def save_classes(self):
        # create txt file for classes
        with open('classes.txt', 'w') as filehandle:
            for user in self.user_dict:
                filehandle.write('%s\n' % user)


if __name__ == "__main__":
    create_dataset = CreateDataset()
