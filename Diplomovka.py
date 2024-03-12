#pip install pillow
import PIL
from PIL import Image
import numpy as np
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



# convered jpg to .png -- JE

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

# zosivenie -- JE

'''
def vytvor_vstupny_vektor_agregacia(obrazok):
    # Konvertujte obrázok na numpy pole
    obrazok_array = np.asarray(obrazok)

    # Rozmery obrázka
    rows, cols = obrazok_array.shape

    # Vytvorenie prazdneho pola pre vstupny vektor
    vstupny_vektor = np.zeros((rows, cols, 8), dtype=np.uint8)

    # Pre každý pixel [x, y] vypočítajte hodnoty |I([x + i, y + j]) - I([x, y])| pre i, j = -1, 0, 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Výpočet hodnôt
            hodnoty = np.abs(obrazok_array - np.roll(obrazok_array, (i, j), axis=(0, 1)))

            # Pridanie hodnôt do vstupného vektora
            vstupny_vektor[:, :, (i + 1) * 3 + (j + 1)] = hodnoty

    return vstupny_vektor

# Cesta k čiernobiélemu obrázku (zosivenému)
cesta_k_ciernobielmu_obrazku = 'C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_zosiveny.png'

# Načítanie čiernobiéleho obrázku
ciernobiely_obrazok = Image.open(cesta_k_ciernobielmu_obrazku)

# Vytvorenie vstupného vektora pre agregáciu
vstupny_vektor_agregacia = vytvor_vstupny_vektor_agregacia(ciernobiely_obrazok)

# Výpis alebo ďalšie spracovanie vstupného vektora
print("Vstupný vektor pre agregáciu:", vstupny_vektor_agregacia)
'''


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
# rozmazanie v matlabe - mozno aj tu, ale gaus nebude


# obrázky pridať a pridať koment ku rozmazaniu

# trojrozmerné pole rxsxf

#[:,:,f]
#imagine_to_array

img_data = PIL.Image.open('C:/Users/zuza8/Desktop/BSR_full/BSR/BSDS500/data/images/test/29030_gradient_rozmazany_obrazok.png')

# Converting the image data into a NumPy array and storing it in 'img_arr'
img_arr = np.array(img_data)

# Printing the NumPy array representation of the image
print(img_arr)

array_pixel = [[-1,-1], [0,-1],[1,-1],[-1,0], [1,0], [-1,1], [0,1], [1,1]]
vector = []

def make_vector(array):
    for i in range(0, len(array)):
        for j in range(0, len(array)):
            for k in range(0, len(array)):
                for l in range(0, len(array_pixel)):
                    x = abs(array[i,j,k] - array[i + array_pixel[l,0], j + array_pixel[l,1],k])
                    vector.append(x)


from itertools import permutations



#Agregačné funkcie
'''
def aritmeticky_priemer(vektor):
    sucet = sum(vektor)
    priemer = sucet / len(vektor)
    return priemer

'''
def choquet_integral(vector, p):
    ch_int = 0.0
    sorted_vector = sorted(vector)
    #pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        ch_int = ch_int + (sorted_vector[i] - sorted_vector[i-1]) * ((len(sorted_vector)-i)/len(vector))**p
    return ch_int

choquet_integral_vysledok = choquet_integral([18,16,10], 0.9)
print(choquet_integral_vysledok)


def CAO_choquet_integral(vector,p, set):
  new_vector=[0]*len(vector)
  for i in range(0, len(set)):
    new_vector[set[i]]=vector[set[i]]
  return choquet_integral(new_vector,p)

print(CAO_choquet_integral([18,16,10],0.9,[0,1,2]))


def aritmeticky_priemer(vektor,p):
    sucet = sum(vektor)
    priemer = sucet / len(vektor)
    return priemer

artm_priem=aritmeticky_priemer([18,16,10],0.9)
print(artm_priem)



def CAO_aritmeticky_priemer(vector,p, set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return aritmeticky_priemer(new_vector,p)

CAO_artm_priem=CAO_aritmeticky_priemer([18,16,10],0.9,[0,1,2])
print(CAO_artm_priem)


def mc_integral(vector, p, function):
    mc_int = 0.0
    auxiliary_vector = vector.copy()
    m_vector = []
    for i in range(0,len(vector)):
        m_vector.insert(0, function(auxiliary_vector))
        auxiliary_vector.pop()

    sorted_vector = sorted(m_vector)
    # pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        mc_int = mc_int + (sorted_vector[i] - sorted_vector[i - 1]) * ((len(sorted_vector) - i) / len(vector)) ** p
    return mc_int


mc_integral_vysledok = mc_integral([18, 16, 10], 0.9, aritmeticky_priemer)
print(mc_integral_vysledok)



def maximum(vector,p):
  return max(vector)


def CAO_maximum(vector,p,set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return maximum(new_vector,p)

print(CAO_maximum([18,16,10],0.9,[1,2]))



def minimum(vector,p):
  return min(vector)


def CAO_minimum(vector,p,set):
  new_vector=[]
  for i in range(0, len(set)):
    new_vector.append(vector[set[i]])
  return minimum(new_vector,p)

print(CAO_minimum([18,16,10],0.9,[0,1]))


def C_Ag_operator(vector, p, array):
  cag=0.0
  auxiliary_vector = []
  for i in range(0,len(vector)):
    func=array[i]
    auxiliary_vector.append(func(vector,p))
  return auxiliary_vector


print(C_Ag_operator([18,16,10],0.9,[aritmeticky_priemer,choquet_integral,minimum]))



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

print(C_APdPid([18,16,10],0.9,[CAO_aritmeticky_priemer,CAO_choquet_integral,CAO_minimum],[[0,1],[0,1,2],[1]]))


# def C_APdPd(vector, p, array, sets): # PI^\downarrow, PHI_\down pre vstupne x:=x^\uparrow
# def C_APuPd(vector, p, array, sets): # PI^\uparrow, PHI_\downarrow pre vstupne x:=x^\uparrow
# def C_APuPu(vector, p, array, sets): # PI^\uparrow, PHI_\id=PHI_\uparrow pre vstupne x:=x^\uparrow

#def a_choquet_integral():

'''
def mc_integral(vector, p, function):
    mc_int = 0.0
    auxiliary_vector = vector.copy()
    m_vector = []
    for i in range(0,len(vector)):
        m_vector.insert(0, function(auxiliary_vector))
        auxiliary_vector.pop()

    sorted_vector = sorted(m_vector)
    # pridať nulu na začiatok
    sorted_vector.insert(0, 0)
    for i in range(1, len(sorted_vector)):
        mc_int = mc_int + (sorted_vector[i] - sorted_vector[i - 1]) * ((len(sorted_vector) - i) / len(vector)) ** p
    return mc_int


mc_integral_vysledok = mc_integral([18, 16, 10], 0.9, aritmeticky_priemer)
print(mc_integral_vysledok)

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




#def c_ag_integral():


#def c_perm_integral(vector, vector_perm_x, vector_perm_ag:

