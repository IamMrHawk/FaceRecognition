# Load libraries
import torch
import cv2
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import torchvision


class FaceIdentification:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # extracting classes
        self.classes = []
        self.read_classes()
        # Transforms
        self.transformer = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        # setting device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.googlenet(pretrained=True)
        self.load_model()
        self.main()

    def read_classes(self):
        with open('classes.txt', 'r') as filehandle:
            for line in filehandle:
                cls = line[:-1]
                self.classes.append(cls)

    def load_model(self):
        # loading trained model
        checkpoint = torch.load('data.pth')
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print('[+] Model Imported successfully')

    def main(self):
        cap = cv2.VideoCapture(0)
        print('[+] Capturing Video')

        while True:
            try:
                _, img = cap.read()

                face = self.face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
                x, y, w, h = face[0][0], face[0][1], face[0][2], face[0][3]
                crpimg = img[y:y + h, x:x + h]

                image = Image.fromarray(crpimg)
                image_tensor = self.transformer(image).float()
                image_tensor = image_tensor.unsqueeze_(0)
                inp = Variable(image_tensor).to(self.device)
                output = self.model(inp).cpu()
                index = output.data.numpy().argmax()
                prediction = self.classes[index]
                print(prediction)

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, prediction, (x, y - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                # img = cv2.resize(img, (500, 500))
                cv2.imshow('img', img)
                cv2.waitKey(1)

                # wait key and termination
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except IndexError:
                continue

            # destroy all open sources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_recognition = FaceIdentification()
