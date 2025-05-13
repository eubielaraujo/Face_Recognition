import cv2
import face_recognition as fr

# Carrega e converte imagens
imgElon = fr.load_image_file('Elon.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = fr.load_image_file('ElonTest1.jpeg')
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# Localiza faces em imagens
faceLoc = fr.face_locations(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

# Marca os pontos faciais 
encodeElon = fr.face_encodings(imgElon)[0]
encodeElonTest = fr.face_encodings(imgElonTest)[0]

# Compara as faces e calcula a distancia de semelhança entre elas
compare = fr.compare_faces([encodeElon],encodeElonTest)
distance = fr.face_distance([encodeElon],encodeElonTest)

# Imprime as informações calculadas acima 
print(compare,distance)

# Executa janelas com as imagens comparadas
cv2.imshow('Elon', imgElon)
cv2.imshow('ElonTest', imgElonTest)
cv2.waitKey(0)