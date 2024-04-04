#pip install pillow
import PIL
from PIL import Image
import numpy as np
from PIL import Image as im
import cv2


def convert_jpg_to_png(input_path, output_path):
    try:
        # Načítanie obrázka z formátu JPEG
        img = Image.open(input_path)

        # Získanie názvu súboru bez prípony
        file_name = input_path.split('.')[0]

        # Vytvorenie cesty pre výstupný súbor vo formáte PNG
        png_output_path = f"{file_name}.png"

        # Konverzia a uloženie obrázka vo formáte PNG
        img.save(png_output_path, 'PNG')

        print(f"Obrázok bol úspešne prekonvertovaný do {png_output_path}")

    except Exception as e:
        print(f"Chyba pri konverzii obrázka: {e}")

# Príklad použitia metódy
convert_jpg_to_png('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030.jpg', 'C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030.png')


def zosiv_obrazok(input_path, output_path):
    try:
        # Načítanie obrázka z formátu PNG
        img = Image.open(input_path)

        # Zosivenie obrázka
        zosiveny_obrazok = img.convert('L')

        # Uloženie zosiveného obrázka
        zosiveny_obrazok.save(output_path)

        print(f"Obrázok bol úspešne zosivený a uložený do {output_path}")

    except Exception as e:
        print(f"Chyba pri zosivovaní obrázka: {e}")

# Príklad použitia metódy
zosiv_obrazok('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030.png', 'C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny.png')


def rozmazanie_gradient1(input_path, output_path):
    try:
        # Načítanie obrázka
        img = cv2.imread(input_path)

        # Aplikovanie Gaussovho rozmazania s kernelom (5, 5) a sigma 2
        rozmazany_obrazok = cv2.GaussianBlur(img, (5, 5), sigmaX=2, sigmaY=2)

        # Uloženie rozmazaného obrázka do výstupného súboru
        cv2.imwrite(output_path, rozmazany_obrazok)

        print(f"Obrázok bol úspešne rozmazaný (gradient) a uložený do {output_path}")

    except Exception as e:
        print(f"Chyba pri rozmazávaní obrázka (gradient): {e}")

rozmazanie_gradient1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny.png', 'C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_gradient_rozmazany_obrazok.png')

def rozmazanie_segmentacia(input_path, output_path):
    try:
        img = cv2.imread(input_path)

        # Metóda na základe segmentácie regiónov
        rozmazany_obrazok = cv2.pyrMeanShiftFiltering(img, 10, 30)

        cv2.imwrite(output_path, rozmazany_obrazok)

        print(f"Obrázok bol úspešne rozmazaný (segmentácia) a uložený do {output_path}")

    except Exception as e:
        print(f"Chyba pri rozmazávaní obrázka (segmentácia): {e}")


rozmazanie_segmentacia('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny.png', 'C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_segmentacia_rozmazany_obrazok.png')

# trojrozmerné pole rxsxf

#[:,:,f]
#imagine_to_array

img_data = PIL.Image.open('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_gradient_rozmazany_obrazok.png')

#img_data1 = PIL.Image.open('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny_GSM(G=0.05,cF=20,t=30).png')

#img_data1 = PIL.Image.open('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny_GSM(G=0.05,cF=70,t=30).png')

img_data1 = PIL.Image.open('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny_anisdiff.png')

# Converting the image data into a NumPy array and storing it in 'img_arr'
img_arr = np.array(img_data, dtype= "int32")

img_arr1 = np.array(img_data1, dtype= "int32")


# Printing the NumPy array representation of the image
#print(img_arr)

print(img_arr1)

#print(img_arr2)

#print(img_arr3)

#print('rozmery', len(img_arr), len(img_arr[0]), len(img_arr[1]))


def choquet_integral(vector, p):
    ch_int = 0.0
    sorted_vector = sorted(vector)
    #pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        ch_int = ch_int + (sorted_vector[i] - sorted_vector[i-1]) * ((len(sorted_vector)-i)/len(vector))**p
    return ch_int

#choquet_integral_vysledok = choquet_integral([18,16,10], 0.9)
#print('Choquetov integral:', choquet_integral_vysledok)

#choquet_integral_vysledok_obr = choquet_integral(vector, 0.9)
#print('Choquetov integral obr:', choquet_integral_vysledok_obr)

def CAO_choquet_integral(vector,p, set):
  new_vector=[0]*len(vector)
  for i in range(0, len(set)):
    new_vector[set[i]]=vector[set[i]]
  return choquet_integral(new_vector,p)

#print('Choquetov integral na podmienenej mnozine:', CAO_choquet_integral([18,16,10],0.9,[0,1,2]))

def aritmeticky_priemer(vektor,p):
    sucet = sum(vektor)
    priemer = sucet / len(vektor)
    return priemer

#artm_priem=aritmeticky_priemer([18,16,10],0.9)
#print('Aritmeticky priemer:', artm_priem)

def CAO_aritmeticky_priemer(vector,p, set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return aritmeticky_priemer(new_vector,p)

#CAO_artm_priem=CAO_aritmeticky_priemer([18,16,10],0.9,[0,1,2])
#print('Aritmeticky priemer na podmienenej mnozine:', CAO_artm_priem)

def mc_integral(vector, p, function):
    mc_int = 0.0
    auxiliary_vector = vector.copy()
    m_vector = []
    for i in range(0,len(vector)):
        m_vector.insert(0, function(auxiliary_vector, p))
        auxiliary_vector.pop()

    sorted_vector = sorted(m_vector)
    # pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        mc_int = mc_int + (sorted_vector[i] - sorted_vector[i - 1]) * ((len(sorted_vector) - i) / len(vector)) ** p
    return mc_int


#mc_integral_vysledok = mc_integral([18, 16, 10], 0.9, aritmeticky_priemer)
#print('MC integral:', mc_integral_vysledok)

def maximum(vector,p):
  return max(vector)

def CAO_maximum(vector,p,set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return maximum(new_vector,p)

#print('Maximum na podmienenej mnozine:', CAO_maximum([18,16,10],0.9,[1,2]))

def minimum(vector,p):
  return min(vector)

def CAO_minimum(vector,p,set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return minimum(new_vector,p)

#print('Minimum na podmienenej mnozine:', CAO_minimum([18,16,10],0.9,[0,1]))


def C_Ag_operator(vector, p, array):
  ag_int=0.0
  auxiliary_vector = []
  for i in range(0,len(vector)):
      func = array[i]
      auxiliary_vector.append(func(vector, p))
  auxiliary_vector = sorted(auxiliary_vector)
  auxiliary_vector.insert(0,0)
  for i in range(1, len(auxiliary_vector)):
      ag_int = ag_int + (auxiliary_vector[i] - auxiliary_vector[i - 1]) * ((len(auxiliary_vector) - i) / len(vector)) ** p
  return ag_int

  return auxiliary_vector

#print('Operator zavisiaci na postupnostiach agregacnych funkci:', C_Ag_operator([18,16,10],0.9,[aritmeticky_priemer,choquet_integral,minimum]))

def C_APdPid(vector, p, array, sets): # PI^\downarrow, PHI_\id=PHI_\uparrow pre vstupne x:=x^\uparrow
  c_app=0.0
  vector.sort()
  auxiliary_vector = []
  for i in range(0,len(vector)):
    cao=array[i]
    auxiliary_vector.append(cao(vector,p,sets[i]))
  auxiliary_vector.sort(reverse=True)
  for i in range(0, len(auxiliary_vector)):
        c_app = c_app + (auxiliary_vector[i])*((((len(auxiliary_vector) - i) / len(vector)) ** p)-(((len(auxiliary_vector) - i-1) / len(vector)) ** p))
  return c_app

#print('Na permutaciach zavisli Choquetov operator: C_APdPid=', C_APdPid([18,16,10],0.9,[CAO_aritmeticky_priemer,CAO_choquet_integral,CAO_minimum],[[0,1],[0,1,2],[1]]))

def C_APdPd(vector, p, array, sets): # PI^\downarrow, PHI_\down pre vstupne x:=x^\uparrow
    c_app = 0.0
    vector.sort(reverse=True)
    auxiliary_vector = []
    for i in range(0, len(vector)):
        cao = array[i]
        auxiliary_vector.append(cao(vector, p, sets[i]))
    auxiliary_vector.sort(reverse=True)
    for i in range(0, len(auxiliary_vector)):
        c_app = c_app + (auxiliary_vector[i]) * ((((len(auxiliary_vector) - i) / len(vector)) ** p) - (
                    ((len(auxiliary_vector) - i - 1) / len(vector)) ** p))
    return c_app

#print('Na permutaciach zavisli Choquetov operator: C_APdPd=', C_APdPd([18, 16, 10], 0.9, [CAO_aritmeticky_priemer, CAO_choquet_integral, CAO_minimum],[[0, 1], [0, 1, 2], [1]]))

def C_APuPd(vector, p, array, sets): # PI^\uparrow, PHI_\downarrow pre vstupne x:=x^\uparrow
    c_app = 0.0
    vector.sort(reverse=True)
    auxiliary_vector = []
    for i in range(0, len(vector)):
        cao = array[i]
        auxiliary_vector.append(cao(vector, p, sets[i]))
    auxiliary_vector.sort()
    for i in range(0, len(auxiliary_vector)):
        c_app = c_app + (auxiliary_vector[i]) * ((((len(auxiliary_vector) - i) / len(vector)) ** p) - (
                    ((len(auxiliary_vector) - i - 1) / len(vector)) ** p))
    return c_app

#print('Na permutaciach zavisli Choquetov operator: C_APuPd=', C_APuPd([18, 16, 10], 0.9, [CAO_aritmeticky_priemer, CAO_choquet_integral, CAO_minimum], [[0, 1], [0, 1, 2], [1]]))

def C_APuPu(vector, p, array, sets): # PI^\uparrow, PHI_\id=PHI_\uparrow pre vstupne x:=x^\uparrow
    c_app = 0.0
    vector.sort()
    auxiliary_vector = []
    for i in range(0, len(vector)):
        cao = array[i]
        auxiliary_vector.append(cao(vector, p, sets[i]))
    auxiliary_vector.sort()
    for i in range(0, len(auxiliary_vector)):
        c_app = c_app + (auxiliary_vector[i]) * ((((len(auxiliary_vector) - i) / len(vector)) ** p) - (
                    ((len(auxiliary_vector) - i - 1) / len(vector)) ** p))
    return c_app

#print('Na permutaciach zavisli Choquetov operator: C_APuPu=', C_APuPu([18, 16, 10], 0.9, [CAO_aritmeticky_priemer, CAO_choquet_integral, CAO_minimum], [[0, 1], [0, 1, 2], [1]]))


'''

def vazeny_aritmeticky_priemer(vektor, vahy):
    return sum(prvek * vaha for prvek, vaha in zip(vektor, vahy)) if len(vektor) != 0 else 0

def mc_integral(vector, p, function, weights):
    mc_int = 0.0
    auxiliary_vector = vector
    m_vector = []

    for i in range(1, len(vector) + 1):
        m_vector.append(function(auxiliary_vector[:i], weights[:i]))

    sorted_vector = sorted(m_vector)
    # pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        mc_int += (sorted_vector[i] - sorted_vector[i - 1]) * ((len(sorted_vector) - i) / len(vector)) ** p
    return mc_int

# Príklad použitia s dvoma vektormi (hodnoty a váhy)
vektor_hodnoty = [18, 16, 10]
vektor_vahy = [3/8, 3/8, 2/8]

vysledok = mc_integral(vektor_hodnoty, 0.9, vazeny_aritmeticky_priemer, vektor_vahy)
print(vysledok)
'''

array_pixel = [[-1,-1], [0,-1],[1,-1],[-1,0], [1,0], [-1,1], [0,1], [1,1]]
sets = [[0, 1, 2, 3, 4, 5, 6, 7], [3, 4], [1, 2, 5, 6], [0, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 6, 7], [1, 2, 5, 6], [0, 1, 2, 5, 6, 7]]

def uprav(x, y):
    pole =[]
    if x < 0:
        x = abs(x)
    if y < 0:
        y = abs(y)
    #if x >= len(img_arr):
    if x >= len(img_arr):
        x = 2*(len(img_arr)-1) - x
    #if y >= len(img_arr[0]):
    if y >= len(img_arr[0]):
        y = 2*(len(img_arr[0])-1) - y
    pole.append(x)
    pole.append(y)
    return pole

def make_vector(i, j):
    vector = []
    for l in range(0, len(array_pixel)):
        posx = i + array_pixel[l][0]
        posy = j + array_pixel[l][1]
        pospole = uprav(posx, posy)
        # if (j + array_pixel[l][1]) > len(array[i]) or (j + array_pixel[l][1]) < 0 or (i + array_pixel[l][0]) < 0 or (i + array_pixel[l][0]) > len(array):
        #    continue
        a = abs(img_arr[i][j][0])
        b = abs(img_arr[pospole[0]][pospole[1]][0])
        w = abs(a-b)
        vector.append(abs(w))
    return vector


#vectoragg = make_vector(320,480)
            #aggvalue = choquet_integral(vectoragg, 0.9)
            #aggvalue = mc_integral(vectoragg, 0.9, aritmeticky_priemer)
#aggvalue = C_Ag_operator(vectoragg, 0.9, [mc_integral, minimum, minimum, minimum, minimum, minimum, minimum, minimum])
#print('hvhsuvuhv', aggvalue)


def feature():
    noveplatno = img_arr.copy()
    noveplatno = noveplatno.astype(np.float64)
    for i in range(0, len(img_arr)):
        for j in range(0, len(img_arr[0])):
            vectoragg = make_vector(i,j)
            #gradient
            aggvalue = choquet_integral(vectoragg, 0.9)
            #aggvalue = mc_integral(vectoragg, 0.9, aritmeticky_priemer)
            #aggvalue = C_Ag_operator(vectoragg, 0.9, [choquet_integral, minimum, aritmeticky_priemer, maximum, choquet_integral, minimum, aritmeticky_priemer, maximum])
            #aggvalue = C_APdPid(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APdPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPu(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)

            noveplatno[i][j][0] = aggvalue
            noveplatno[i][j][1] = aggvalue
            noveplatno[i][j][2] = aggvalue

    #img = Image.fromarray(noveplatno, 'RGB')
    img = im.fromarray(noveplatno.astype(np.uint8))
    img.show()

    #gradient
    img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APuPu_obr.png')


#feature()

def uprav1(x, y):
    pole =[]
    if x < 0:
        x = abs(x)
    if y < 0:
        y = abs(y)
    if x >= len(img_arr1):
        x = 2*(len(img_arr1)-1) - x
    #if y >= len(img_arr[0]):
    if y >= len(img_arr1[0]):
        y = 2*(len(img_arr1[0])-1) - y
    pole.append(x)
    pole.append(y)
    return pole

def make_vector1(i, j):
    vector = []
    for l in range(0, len(array_pixel)):
        posx = i + array_pixel[l][0]
        posy = j + array_pixel[l][1]
        pospole = uprav1(posx, posy)
        # if (j + array_pixel[l][1]) > len(array[i]) or (j + array_pixel[l][1]) < 0 or (i + array_pixel[l][0]) < 0 or (i + array_pixel[l][0]) > len(array):
        #    continue
        a = abs(img_arr1[i][j])
        b = abs(img_arr1[pospole[0]][pospole[1]])
        w = abs(a-b)
        vector.append(abs(w))
    return vector

def feature1():
    noveplatno = img_arr1.copy()
    noveplatno = noveplatno.astype(np.float64)
    for i in range(0, len(img_arr1)):
        for j in range(0, len(img_arr1[0])):
            vectoragg = make_vector1(i, j)
            # GSM(G=0.05,cF=20,t=30)
            #aggvalue = choquet_integral(vectoragg, 0.9)
            #aggvalue = mc_integral(vectoragg, 0.9, aritmeticky_priemer)
            #aggvalue = C_Ag_operator(vectoragg, 0.9, [choquet_integral, minimum, aritmeticky_priemer, maximum, choquet_integral, minimum, aritmeticky_priemer, maximum])
            #aggvalue = C_APdPid(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APdPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            aggvalue = C_APuPu(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum,CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer,CAO_maximum], sets)

            noveplatno[i][j] = aggvalue
            noveplatno[i][j] = aggvalue

    # img = Image.fromarray(noveplatno, 'RGB')
    img = im.fromarray(noveplatno.astype(np.uint8))
    img.show()

    # 29030_zosiveny_GSM(G=0.05,cF=20,t=30)
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APuPd_obr.png')
    img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APuPu_obr.png')

#feature1()

def feature2():
    noveplatno = img_arr1.copy()
    noveplatno = noveplatno.astype(np.float64)
    for i in range(0, len(img_arr1)):
        for j in range(0, len(img_arr1[0])):
            vectoragg = make_vector1(i,j)
            #GSM(G=0.05,cF=70,t=30)
            aggvalue = choquet_integral(vectoragg, 0.9)
            #aggvalue = mc_integral(vectoragg, 0.9, aritmeticky_priemer)
            #aggvalue = C_Ag_operator(vectoragg, 0.9, [choquet_integral, minimum, aritmeticky_priemer, maximum, choquet_integral, minimum, aritmeticky_priemer, maximum])
            #aggvalue = C_APdPid(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APdPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPu(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum,CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer,CAO_maximum], sets)

            noveplatno[i][j] = aggvalue
            noveplatno[i][j] = aggvalue

    #img = Image.fromarray(noveplatno, 'RGB')
    img = im.fromarray(noveplatno.astype(np.uint8))
    img.show()

    # 29030_zosiveny_GSM(G=0.05,cF=70,t=30)
    img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APuPu_obr.png')

#feature2()


def feature3():
    noveplatno = img_arr1.copy()
    noveplatno = noveplatno.astype(np.float64)
    for i in range(0, len(img_arr1)):
        for j in range(0, len(img_arr1[0])):
            vectoragg = make_vector1(i,j)
            #anisdiff
            #aggvalue = choquet_integral(vectoragg, 0.9)
            #aggvalue = mc_integral(vectoragg, 0.9, aritmeticky_priemer)
            #aggvalue = C_Ag_operator(vectoragg, 0.9, [choquet_integral, minimum, aritmeticky_priemer, maximum, choquet_integral, minimum, aritmeticky_priemer, maximum])
            #aggvalue = C_APdPid(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APdPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            #aggvalue = C_APuPd(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum, CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum], sets)
            aggvalue = C_APuPu(vectoragg, 0.9, [CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer, CAO_maximum,CAO_choquet_integral, CAO_minimum, CAO_aritmeticky_priemer,CAO_maximum], sets)

            noveplatno[i][j] = aggvalue
            noveplatno[i][j] = aggvalue

    #img = Image.fromarray(noveplatno, 'RGB')
    img = im.fromarray(noveplatno.astype(np.uint8))
    img.show()

    # anisdiff
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APuPd_obr.png')
    img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APuPu_obr.png')

#feature3()

def scaling(image):
    min = 256
    max = 0
    img_convert = PIL.Image.open(image)
    img_convert_arr = np.array(img_convert, dtype="int32")

    for i in range(0, len(img_convert_arr)):
        for j in range(0, len(img_convert_arr[0])):
            if img_convert_arr[i][j][0] > max:
                max = img_convert_arr[i][j][0]
            if img_convert_arr[i][j][0] < min:
                min = img_convert_arr[i][j][0]

    const = 255/(max-min)

    for i in range(0, len(img_convert_arr)):
        for j in range(0, len(img_convert_arr[0])):
            img_convert_arr[i][j] = (img_convert_arr[i][j] - min)*const

    img = im.fromarray(img_convert_arr.astype(np.uint8))
    #img.show()

    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient_scaling/29030_scaling_C_APuPu_obr.png')

#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_choquetobr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_mc_int_obr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_Ag_operator_obr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APdPid_obr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APdPd_obr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APuPd_obr.png')
#scaling('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_gradient/29030_C_APuPu_obr.png')




def scaling1(image):
    min = 256
    max = 0
    img_convert = PIL.Image.open(image)
    img_convert_arr = np.array(img_convert, dtype="int32")

    for i in range(0, len(img_convert_arr)):
        for j in range(0, len(img_convert_arr[0])):
            if img_convert_arr[i][j] > max:
                max = img_convert_arr[i][j]
            if img_convert_arr[i][j] < min:
                min = img_convert_arr[i][j]

    const = 255 / (max - min)
    for i in range(0, len(img_convert_arr)):
        for j in range(0, len(img_convert_arr[0])):
            img_convert_arr[i][j] = (img_convert_arr[i][j] - min) * const

    img = im.fromarray(img_convert_arr.astype(np.uint8))
    img.show()

    # 29030_zosiveny_GSM(G=0.05,cF=20,t=30)
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_scaling_C_APuPu_obr.png')

    # 29030_zosiveny_GSM(G=0.05,cF=70,t=30)
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_scaling_C_APuPu_obr.png')

    # anisdiff
    img.save('Output/29030_zosiveny_anisdiff_scaling_choquetobr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_mc_int_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_C_Ag_operator_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_C_APdPid_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_C_APdPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_C_APuPd_obr.png')
    #img.save('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff_scaling/29030_zosiveny_anisdiff_scaling_C_APuPu_obr.png')

# 29030_zosiveny_GSM(G=0.05,cF=20,t=30)
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_choquetobr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_mc_int_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_Ag_operator_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APdPid_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APdPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APuPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=20,t=30)/29030_zosiveny_GSM(G=0.05,cF=20,t=30)_C_APuPu_obr.png')


# 29030_zosiveny_GSM(G=0.05,cF=70,t=30)
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_choquetobr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_mc_int_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_Ag_operator_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APdPid_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APdPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APuPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_GSM_29030_zosiveny_GSM(G=0.05,cF=70,t=30)/29030_zosiveny_GSM(G=0.05,cF=70,t=30)_C_APuPu_obr.png')


# anisdiff
scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_choquetobr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_mc_int_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_Ag_operator_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APdPid_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APdPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APuPd_obr.png')
#scaling1('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/Auto_zosiveny_anisdiff/29030_zosiveny_anisdiff_C_APuPu_obr.png')