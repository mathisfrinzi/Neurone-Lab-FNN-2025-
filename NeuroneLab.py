# -*- coding: utf-8 -*-
"""
@author: mathf
"""

from tkinter import *
from tkinter import scrolledtext
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.messagebox
import random

def normalisation(x):
    '''Fonction de normalisation standard : on retranche par la moyenne et on divise par l'écart-type de l'échantillon x en entrée'''
    x = x.astype(float)
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

def normalisation_min_max(x):
    '''Fonction de normalisation min-max'''
    x = x.astype(float)
    mu = np.mean(x,0)
    sigma = x.max() - x.min()
    x_norm = (x-mu)/sigma
    return x_norm, mu, sigma

def normalisation_max(x):
    '''Fonction de normalisation max : les valeurs de sortie sont entre -1 et 1, -1 correspond au minimum et 1 correspond au maximum de l'échantillon d'entrée x'''
    x = x.astype(float)
    mu = np.mean(x,0)
    x_norm = x-mu
    sigma = max(x_norm.max(),-x_norm.min())
    x_norm = x_norm/sigma
    return x_norm,mu,sigma

def tanh(z, deriv=False):
    '''Renvoie tanh(z) ou tanh'(z) si deriv=True'''
    t = np.tanh(z)
    return (1 - t**2) if deriv else t

def relu(z, deriv=False):
    '''Renvoie relu(z) ou relu'(z) si deriv = True'''
    r = np.zeros(z.shape)
    if deriv:
        pos = np.where(z>=0)
        r[pos] = 1.0
        return r
    else :    
        return np.maximum(r,z)

def leaky_relu(z, deriv=False, alpha=0.01):
    '''Renvoie leaky_relu(z) ou leaky_relu'(z) si deriv=True
    Le coefficient alpha permet d'adapter le leaky_relu
    '''
    if deriv:
        return np.where(z > 0, 1, alpha)
    return np.where(z > 0, z, alpha * z)


def softmax(a, deriv = False):
    '''Renvoie softmax(z) si deriv=False
    Si deriv=True, la fonction renvoie 1 et le programme se comportera comme si deltah = deltaa dans la passe-arrière'''
    a_ = a.copy()
    K = a_.shape[0]
    somme = sum([np.exp(a_[i]) for i in range(K)])
    s = np.exp(a_)/somme
    if not deriv:
        return s
    return 1 #on fait ça pour que le deltah soit égal à deltaa

def sigmoide(z, deriv=False):
    '''Renvoie sigmoide(z) ou sigmoide'(z) si deriv=True'''
    s = 1 / (1 + np.exp(-z))
    if deriv:
        return s * (1 - s)
    else :
        return s
    
def lineaire(z, deriv=False):
    '''Renvoie lineaire(z) ou lineaire'(z) si deriv=True'''
    if deriv:       
        return 1     
    else :
        return z

def calcule_cout_mse(y, d):
    '''Renvoie la moyennes des différences au carré entre y et d''' 
    cout = np.sqrt(np.mean((y-d)**2))
    return cout

mse = calcule_cout_mse
MSE = mse

def calcule_cout_lineaire(X, Y, theta):
    N,nb_var = X.shape
    cout = ((theta.T.dot(X)-Y)**2)
    try:
        return sum(cout[0])/(2*N)
    except TypeError:
        return sum(cout)/(2*N)

def calcule_cout_entropie_croisee( a,d, epsilon =10**(-6)):
    cout = -np.sum(d * np.log(abs(a[-1])+epsilon)) / d.shape[0]
    return cout

entropie_croisee = calcule_cout_entropie_croisee

def passe_avant(x, W, b, activation):
    '''Algorithme de passe-avant, en fonction des poids contenus dans W et b et de la structure du réseau contenu dans activation, on obtient à partir de l'entrée x la sortie du réseau de neurone'''
    h = [x]
    a = []
    for i in range(len(b)):
        value = W[i]@h[-1]+b[i]
        a.append(value)
        g = activation[i]
        h.append(g(value))
    h.append(activation[-1](h[-1]))
    return a, h

def passe_arriere(delta_h, a, h, W, activation):
    '''Passe-arrière : permet de corriger le réseau en fonction de l'erreur sur la sortie contenue dans delta_h
    La correction est contenue dans delta_W et delta_b.    '''
    delta_b = []
    delta_W = []
    delta = delta_h.copy()

    for i in range(len(W)-1, -1, -1):
        delta_a = delta*activation[i](a[i], True)
        delta_b.append(delta_a.mean(1).reshape(-1, 1))
        delta_W.append(delta_a.dot(h[i].T) / h[i].shape[1])
        delta_h = (W[i].T).dot(delta_a)
    
    delta_b = delta_b[::-1]
    delta_W = delta_W[::-1]

    return delta_W, delta_b

RN_classes = {}
RN_classes_classes_norm = {}

def reseau_neurone_classification(act,normalisation = normalisation, classe = None):
    classe_test = classe
    for i in entree_sortie[1]:
        selection_entr = entr0[:,i]
        selection_val = val0[:,i]
        selection_test = test0[:,i]
        classes = list(set(list(selection_entr)+list(selection_val)+list(selection_test)))
        if classe_test == None:
            classe_test = classes[0]
        for j in range(selection_entr.shape[0]):
            entr0[j,i] = int(classe_test==entr0[j,i])
        for j in range(selection_val.shape[0]):
            val0[j,i] = int(classe_test==val0[j,i])
        for j in range(selection_test.shape[0]):
            test0[j,i] = int(classe_test==test0[j,i])
            
    H = (reseau_neurone(act, calcule_cout_entropie_croisee, normalisation, classes=classes), classes)
    l_hyper = liste_hyper.copy()
    for i in range(len(l_hyper)):
        l_hyper[i] = eval(l_hyper[i])
    t = tuple(l_hyper)
    
    return H 
    
def reseau_neurone_regression(activation, calcule_cout, normali, entree_sortie, alpha, n_iters, D_c, numero=0):
    return reseau_neurone(activation, calcule_cout, normali, entree_sortie, alpha, n_iters, D_c, numero=0)

RN_cout_app = {}
RN_cout_val = {}
RN_cout_test = {}
RN_resultats = {}
COURBES = {}
liste_hyper = ['alpha','n_iters','Ncouches','epaisseurCouches']
def reseau_neurone(activation, calcule_cout, normali, entree_sortie, alpha, n_iters, D_c, numero=0,classes = None):
    
    for i in range(len(entree_sortie[0])):
        COURBES['X{0}'.format(i)] = list(data0[:, entree_sortie[0][i]].astype('float'))
        COURBES['Xtest{0}'.format(i)] = list(test0[:,entree_sortie[0][i]].astype('float'))
    
    for i in range(len(entree_sortie[1])):
        COURBES['Y{0}'.format(i)] = data0[:, entree_sortie[1][i]]
        COURBES['Ytest{0}'.format(i)] = test0[:,entree_sortie[1][i]]
    
    nb_iters = n_iters
    nb_var = len(entree_sortie[0])
    nb_cible = len(entree_sortie[1])
        
    x_app = entr0[:,entree_sortie[0]].copy()
    d_app = entr0[:,entree_sortie[1]].copy()
    x_val = val0[:,entree_sortie[0]].copy()
    d_val = val0[:,entree_sortie[1]].copy()
    x_test = test0[:,entree_sortie[0]].copy()
    d_test = test0[:,entree_sortie[1]].copy()
    
    x_app = x_app.astype(float)
    x_val = x_val.astype(float)
    x_test = x_test.astype(float)
    d_app = d_app.astype(float)
    d_val = d_val.astype(float)
    d_test = d_test.astype(float)
    
    x = np.concatenate((x_app,x_val,x_test),axis=0)
    x,mu,sigma = normali(x)
    x_app = (x_app-mu)/sigma
    x_val = (x_val-mu)/sigma
    x_test = (x_test-mu)/sigma
    
    d_app, mu_d, sigma_d = normalisation_max(d_app)
    d_val = (d_val - mu_d) / sigma_d
    d_test = (d_test - mu_d) / sigma_d
    
    couts_apprentissage = np.zeros(nb_iters)
    couts_validation = np.zeros(nb_iters)
    
    # Initialisation aléatoire des poids du réseau
    W = []
    b = []
    for i in range(len(D_c)-1):    
        W.append(np.random.randn(D_c[i+1], D_c[i]) * np.sqrt(2 / D_c[i]))
        b.append(np.zeros((D_c[i+1],1)))

    x_app = x_app.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
    d_app = d_app.T 

    x_val = x_val.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
    d_val = d_val.T 

    x_test = x_test.T # Les données sont présentées en entrée du réseau comme des vecteurs colonnes
    d_test = d_test.T
    for t in range(nb_iters):
        a, h = passe_avant(x_val, W, b, activation)
        y_val = h[-1] 
        a, h = passe_avant(x_app, W, b, activation)
        y_app = h[-1] 
        
        couts_apprentissage[t] = calcule_cout(y_app,d_app)
        couts_validation[t] = calcule_cout(y_val,d_val)
        delta_h = (y_app-d_app)  
        delta_W, delta_b = passe_arriere(delta_h, a, h, W, activation)
        for i in range(len(b)-1,-1,-1):
            W[i] -= alpha * delta_W[i]
            b[i] -= alpha * delta_b[i]
    
    COURBES['couts_apprentissage(t)'] = couts_apprentissage
    COURBES['couts_validation(t)'] = couts_validation
    
    a, h = passe_avant(x_test, W, b, activation)
    y_test = h[-1] # Sortie prédite normalisée
    cout = calcule_cout(y_test,d_test)
    COURBES['couts_test'] = [cout for t in range(nb_iters)]
    
    for i in range(len(entree_sortie[1])):
        COURBES['Ytest_pred{0}'.format(i)] = (sigma_d*y_test+mu_d).T
    
    COURBES['t'] = [t for t in range(nb_iters)]
    
    return [(W,b,activation,mu,sigma,mu_d,sigma_d), cout]
    #l_hyper = liste_hyper.copy()
    for i in range(len(l_hyper)):
        l_hyper[i] = eval(l_hyper[i])
    t = tuple(l_hyper)
    RN_cout_app[t] = couts_apprentissage
    RN_cout_val[t] = couts_validation
    
    
    RN_cout_test[t] = cout
    if classes != None:
        RN_classes[t] = [(i-mu_d)/sigma_d for i in range(len(classes))]
    RN_resultats[t] = (W,b,activation,mu,sigma,mu_d,sigma_d)

    return RN_resultats[t]


def datavide():
    return np.array([[]])
data1 = datavide()
data2 = datavide()
data0 = datavide()

COURBES = {}

class Interface(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.mainPaned = PanedWindow(self, orient = VERTICAL)
        self.dataPaned = PanedWindow(self, orient = HORIZONTAL)
        self.title('Interface pour un réseau de neurones feedforward')
        paned_collect_data = PanedWindowData(self,0)
        self.dataPaned.add(paned_collect_data)
        self.dataPaned.add(paned_collect_data.text_area)
        self.dataPaned.add(PanedWindowDivision(self,0))
        self.mainPaned.add(self.dataPaned)
        
        
        canvas = CanvasRN(self)
        self.canvas = canvas
        self.mainPaned.add(canvas)
        self.mainPaned.pack()
        
        self.importantPaned = PanedWindow(self,orient=VERTICAL, bg='orange')
        self.importantPaned.add(Label(self, text="Nombre d'itérations : ", bg='orange',anchor="w"))
        self.entree_n_iters = Entry(self)
        self.entree_n_iters.insert(0,"1000")
        self.importantPaned.add(self.entree_n_iters)
        self.importantPaned.add(Label(self, text="Coefficient d'apprentissage : ", bg='orange',anchor="w"))
        self.entree_alpha = Entry(self)
        self.entree_alpha.insert(0,'0.01')
        self.importantPaned.add(self.entree_alpha)
        self.importantPaned.add(Label(self,text="Indices d'entrée/sortie :", bg='orange',anchor="w"))
        self.entree_entree_sortie = Entry(self)
        self.entree_entree_sortie.insert(0,"[[0],[1]]")
        self.importantPaned.add(self.entree_entree_sortie)
        self.importantPaned.add(Label(self,text="Fonction cout :", bg='orange',anchor="w"))
        self.entree_calcule_cout = Entry(self)
        self.entree_calcule_cout.insert(0,"calcule_cout_mse")
        self.importantPaned.add(self.entree_calcule_cout)
        self.dataPaned.add(self.importantPaned)
                
        self.importantPaned.add(Button(self,text="Entraîner le réseau", command=self.regression))
        self.importantPaned.add(Button(self,text="Tracer des courbes", command=self.open_courbe))
        
        # Entrée et sortie
        self.liste_couche_entree = [CanvasElement(self.canvas, 'rectangle', [0,0,0,0],fill = 'light green')
        , CanvasElement(self.canvas, 'text', [0,0,0,0], text = '')
        , CanvasElement(self.canvas, 'rectangle', [0,0,0,0],fill = 'light green')
        , CanvasElement(self.canvas, 'text', [0,0,0,0], text = '')]
                                    
        self.couches = [Couche(canvas,1,0,leaky_relu), Couche(canvas, 5, 1, tanh),Couche(canvas, 1,2,lineaire)]

        self.testPaned = PanedWindow(self,orient=HORIZONTAL,bg="pink")
        self.entree_test = Entry(self)
        self.entree_test.insert(0,'[[0]]')
        self.entree_test.pack()
        self.testPaned.add(self.entree_test)
        self.testPaned.add(Button(self,command=self.tester, text="Tester le réseau pour cette entrée"))
        self.mainPaned.add(self.testPaned)
        
        tbut = 20
        self.canvas.add_button_canvas('<Button-1>', self.zoomer, 10,-10,10+tbut,-10-tbut, apparent=(True,{'fill':'purple'}))
        self.canvas.add_button_canvas("<Button-1>", self.dezoomer, 10+3*tbut,-10,10+4*tbut,-10-tbut, apparent=(True, {'fill':'green'}))
        self.sortie_test = "Lancez une simulation"
        
        self.after_()
        self.mainloop()
    def zoomer(self):
        global ZOOM
        ZOOM *= 2 
    def dezoomer(self):
        global ZOOM
        ZOOM /= 2
    def tester(self):
        (W,b,activation,mu,sigma,mu_d,sigma_d) = self.result
        x = np.array(eval(self.entree_test.get()))
        x = (x-mu)/sigma
        x=x.T
        self.couches[0].texte = np.array(x)
        a,h = passe_avant(x,W,b,activation)
        for i in range(len(a)):
            self.couches[i+1].texte = activation[i](a[i])
        y = (sigma_d*h[-1]+mu_d).T
        self.sortie_test = y
        return y
    def open_courbe(self):
        InterfaceCourbe()
    def regression(self):
        activation = [i.activation for i in self.couches]
        D_c = [i.nombre for i in self.couches]
        entree_sortie = eval(self.entree_entree_sortie.get())
        calcule_cout = eval(self.entree_calcule_cout.get())
        n_iters = eval(self.entree_n_iters.get())
        alpha = eval(self.entree_alpha.get())
        if type(alpha) == type([]):
            _ = alpha.copy()
            COURBES["alpha"] = alpha
            l = []
            for alph in alpha:
                result, cout = reseau_neurone_regression( activation, calcule_cout, normalisation, entree_sortie,alph, n_iters, D_c  )
                l.append(cout)
            COURBES['cout(alpha)'] = l
        
        else:
            alph = alpha
            result, cout = reseau_neurone_regression( activation, calcule_cout, normalisation, entree_sortie,alph, n_iters, D_c  )

            if 'alpha' in COURBES.keys():
                del COURBES['alpha']
        self.result = result
    def after_(self):
        for couche in self.couches:
            couche.update_canvas()
        
        taille = Couche.WIDTH//2
        x,y = (-Couche.WIDTH//2, 40+Couche.HEIGHT//2)
        self.liste_couche_entree[0].update_coords([x-taille, y-taille, x+taille, y+taille] )
        self.liste_couche_entree[1].update_coords([x,y])
        try:
            self.liste_couche_entree[1].kwargs['text'] = 'In : '+str(self.entree_test.get())+'\n µ = {0}\n sigma = {1}'.format(self.result[3],self.result[4])
        except:
            self.liste_couche_entree[1].kwargs['text'] = 'In : '+str(self.entree_test.get())
        n = len(self.couches)
        x,y = (Couche.WIDTH*n+Couche.WIDTH//2, 40+Couche.HEIGHT//2)
        self.liste_couche_entree[2].update_coords([x-taille, y-taille, x+taille, y+taille] )
        self.liste_couche_entree[3].update_coords([x,y])
        try:
            self.liste_couche_entree[3].kwargs['text'] = 'Out : '+str(self.sortie_test)+'\n µ_d = {0}\n sigma_d = {1}'.format(self.result[5],self.result[6])
        except:
            pass
        try:
            l = eval(self.entree_entree_sortie.get())
            self.couches[0].nombre = len(l[0])
            self.couches[-1].nombre = len(l[1])
        except:
            pass
        self.after(10,self.after_)

class InterfaceCourbe(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.courbe0 = list(COURBES.keys()).copy()
        self.courbe1 = self.courbe0.copy()
        self.selected = []
        self.mainPaned = PanedWindow(self, orient = HORIZONTAL, height = 600)
        
        self.listebtn1 = []
        self.btnPaned0 = PanedWindow(self,orient=VERTICAL)
        self.btnPaned0.add(Label(self, text="Ordonnée(s) : "))
        for i in self.courbe0:
            b = Button(self,text=i, command = lambda x=i:self.definir1(x))
            self.listebtn1.append(b)
            self.btnPaned0.add(b)
        
        self.mainPaned.add(self.btnPaned0)
        
        self.listebtn2 = []
        self.btnPaned1 = PanedWindow(self,orient=VERTICAL)
        self.btnPaned1.add(Label(self,text="Abscisse :"))
        for i in self.courbe1:
            b = Button(self,text=i, command = lambda x=i:self.definir2(x))
            self.listebtn2.append(b)
            self.btnPaned1.add(b)
        
        self.mainPaned.add(self.btnPaned1)
        
        self.mainPaned.add(Button(self,text="Tracez la courbe", command = self.definir))
        self.canvas_cout = None
        self.mainPaned.pack()
        self.mainloop()
    
    def definir(self):
        fig, ax = plt.subplots()
        y = self.element2
        for x in self.selected:
            X = COURBES[x]
            Y = COURBES[y]
            ax.plot(Y,X,marker="x",linestyle="", label='{0}({1})'.format(x,y))
        plt.legend()
        if self.canvas_cout != None:
            self.mainPaned.forget(self.last_canvas)
        self.canvas_cout = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = self.canvas_cout.get_tk_widget()
        self.last_canvas = canvas_widget
        self.mainPaned.add(canvas_widget)
    
    def definir1(self, element):
        for b in self.listebtn1:
            b['bg'] = 'red'
            if b['text'] in self.selected:
                b['bg'] = 'light green'
            if b['text'] == element:
                if b["text"] in self.selected:
                    b['bg'] = 'red'
                    self.selected.remove(element)
                else:
                    b['bg'] = "light green"
                    self.selected.append(element)
        self.element1 = element
        
    def definir2(self,element):
        for b in self.listebtn2:
            b['bg'] = 'red'
            if b['text'] == element:
                b['bg'] = 'light green'
        self.element2 = element

class InterfaceCouche(Tk):
    def __init__(self, couche):
        Tk.__init__(self)
        self.couche = couche
        
        self.mainPaned = PanedWindow(self,orient=VERTICAL)
        self.mainPaned.add(Label(self, text="Nombre de neurones pour cette couche :"))
    
        self.entree_n_neurone = Entry(self)
        self.entree_n_neurone.insert(0,"{0}".format(couche.nombre))
        self.mainPaned.add(self.entree_n_neurone)
        self.mainPaned.add(Label(self, text="Fonctions d'activation :"))
        self.entree_activation = Entry(self)
        self.entree_activation.insert(0,couche.activation.__name__)
        self.mainPaned.add(self.entree_activation)
        self.mainPaned.add(Button(self,text="Appliquer", command = self.appliquer))
        
        self.mainPaned.pack()
        self.mainloop()
        self.couche.open_interface = True
    def appliquer(self):
        self.couche.nombre = eval(self.entree_n_neurone.get())
        self.couche.activation = eval(self.entree_activation.get())
        self.destroy()
        self.couche.open_interface = True
    

ZOOM = 0.4

class Couche:
    HEIGHT = 800
    WIDTH = 400
    R = 20
    COUCHE = {}
    def __init__(self, canvas, nombre, numero_couche, activation):
        Couche.WIDTH = max(Couche.WIDTH, Couche.R*5*nombre)
        self.nombre = nombre
        self.canvas = canvas
        self.numero_couche = numero_couche
        self.elements = []
        self.elements_ligne = []
        self.activation = activation
        self.texte = ['']
        Couche.COUCHE[self.numero_couche] = self
        #éléments graphiques associés au couche
        self.id_bouton = None 
        self.update_position_bouton()
        self.interface_open = True
    def recheck_size(self):
        
        Couche.HEIGHT = int(max([ZOOM*Couche.R*5*Couche.COUCHE[i].nombre for i in Couche.COUCHE.keys()]))
        Couche.WIDTH = 400*ZOOM
        self.update_position_bouton()
    def update_position_bouton(self):
        taille = Couche.WIDTH//5//2
        x = Couche.WIDTH*self.numero_couche + Couche.WIDTH//2-Couche.WIDTH//4
        if True:
            if self.id_bouton == None:
                self.id_bouton = [self.canvas.add_button_canvas('<Button-1>', self.ajout_couche,x,10,x+taille,10+taille, apparent=(True,{'fill':'red','text':'+'})) 
                                  ,self.canvas.add_button_canvas('<Button-1>', self.modifier_valeur, x+Couche.WIDTH//4, 10, x+taille+Couche.WIDTH//4, 10+taille, apparent=(True,{"fill":"pink"}) )] 
            else:
                self.id_bouton[0].update_coords( [x,10,x+taille,10+taille])
                self.id_bouton[1].update_coords([x+Couche.WIDTH//4, 10, x+taille+Couche.WIDTH//4, 10+taille])
                
    def changer_numero(self,new_numero):
        del Couche.COUCHE[self.numero_couche]
        Couche.COUCHE[new_numero] = self
        self.numero_couche = new_numero
    def modifier_valeur(self):
        if True:
            self.interface_open = False
            InterfaceCouche(self)
            self.interface_open = True
    def ajout_couche(self):
        #ajout_couche à gauche
        c = Couche.COUCHE.copy()
        n = self.numero_couche
        m = list(c.keys())
        m.sort(reverse=True)
        for i in m:
            if i>=n:
                c[i].changer_numero(i+1)
        Couche(self.canvas, 1, n, self.activation)
        self.canvas.master.couches.insert(n,Couche.COUCHE[n])
        for i in Couche.COUCHE.keys():
            Couche.COUCHE[i].update_position_bouton()
    def get_coordonnees_noeuds(self):
        coordonnees = []
        x = Couche.WIDTH//3 + Couche.WIDTH*self.numero_couche
        for i in range(self.nombre):
            y = Couche.HEIGHT//(self.nombre+2)*(i+1) +50
            coordonnees.append((x,y))
        return coordonnees
    def update_canvas(self):
        self.recheck_size()
        if self.numero_couche != 0:
            coord1 = Couche.COUCHE[self.numero_couche-1].get_coordonnees_noeuds()
            coord2 = self.get_coordonnees_noeuds()
            while len(self.elements_ligne) <= (len(coord1))*(len(coord2)):
                self.elements_ligne.append([CanvasElement(self.canvas,'line',[0,0,0,0])])
            for i in range(len(coord1)):
                x0,y0 = coord1[i]
                for j in range(len(coord2)):
                    x1,y1= coord2[j]
                    self.elements_ligne[i*len(coord2)+j][0].update_coords([x0,y0,x1,y1])
            while len(self.elements_ligne) > (len(coord1)*len(coord2)):
                el = self.elements_ligne.pop()
                el[0].update_coords([0,0,0,0])
                
                
        while len(self.elements)<self.nombre:
            self.elements.append([CanvasElement(self.canvas,'oval',[0,0,0,0], fill = 'red'),
                                  CanvasElement(self.canvas,'text',[0,0],text='', anchor="w"),
                                  CanvasElement(self.canvas,'text',[0,0],fill='blue',text='', anchor="w")
                                  ])
        coords = self.get_coordonnees_noeuds()
        for i in range(self.nombre):
            x,y = coords[i]
            self.elements[i][0].update_coords([x-Couche.R//2,y-Couche.R//2, x+Couche.R//2, y+Couche.R//2 ])
            self.elements[i][1].kwargs['text'] = self.activation.__name__
            self.elements[i][1].update_coords([x+10,y+10])
            if i < len(self.texte):
                self.elements[i][2].kwargs['text'] = self.texte[i]
                self.elements[i][2].update_coords([x+10,y-10])
        while len(self.elements)> self.nombre:
            el = self.elements.pop()
            el[0].update_coords([0,0,0,0])
            el[1].update_coords([0,0])
            el[2].update_coords([0,0])
        
class ButtonCanvas:
    def __init__(self ,canvas, type_clic, command, x0,y0,x1,y1, apparent = (False,{})):
        self.canvas = canvas
        c = None
        self.elements=[]
        kwargs = apparent[1]
        if 'text' in kwargs.keys():
            t = kwargs['text']
            del kwargs['text']
            coord = [(x0+x1)//2, (y0+y1)//2]
        if apparent[0]:
            c = CanvasElement(self.canvas,'rectangle',[x0,y0,x1,y1], **kwargs)
        x0,x1 = min(x0,x1),max(x0,x1)
        y0,y1 = min(y0,y1),max(y0,y1)
        info = (x0,y0,x1,y1, command)
        self.info = info
        self.element = c
        self.elements.append(c)
        try:
            self.elements.append(CanvasElement(self.canvas, 'text', coord, text=t))
        except:
            pass
    def update_coords(self,args,**kwargs):
        self.info = (args[0],args[1],args[2],args[3], self.info[4])
        for i in self.elements:
            if i.type_dessin =='text':
                arg = [(args[0]+args[2])//2, (args[1]+args[3])//2]
                i.update_coords(arg,**kwargs)
            else:
                i.update_coords(args,**kwargs)

class CanvasRN(Canvas):
    def __init__(self,*args,**kwargs):
        kwargs['bg'] = 'white'
        kwargs['height'] = 400
        kwargs['width'] = 1200
        CanvasElement.liste_element[self] = []
        Canvas.__init__(self,*args,**kwargs)
        self.bind('<Motion>',self.motion)
        self.c = CanvasElement(self, 'oval', [0,0,10,10])
        self.dX, self.dY = (0,100)
        self.liste_button_canvas = []
        self.bind('<Button-1>',lambda e, x="<Button-1>": self.clic(e,x))
        self.after()
    def clic(self, e, bouton='<Button-1>'):
        for i in self.liste_button_canvas:
            x0,y0,x1,y1,command = i.info
            if x0<=e.x - self.dX<=x1 and y0<=e.y-self.dY<=y1:
                command()
                break
    def add_button_canvas(self,type_clic, command, x0,y0,x1,y1, apparent = (False,{})):
        b= ButtonCanvas(self,type_clic, command, x0, y0, x1, y1, apparent)
        self.liste_button_canvas.append(b)
        return b
    def after(self):
        for i in self.c.get_liste(self):
            i.update()
            i.decalageX = self.dX
            i.decalageY = self.dY
        self.master.after(10,self.after)
    def add_element(self,type_element, coords:list, **kwargs):
        CanvasElement(self,type_element,coords,**kwargs)
        
    def motion(self,e):
        x,y = e.x, e.y
        portee = 20
        sensi = 30
        if x < portee:
            self.decalageX(sensi)
        if y < portee:
            self.decalageY(sensi)
        if y > int(self['height'])-portee:
            self.decalageY(-sensi)
        if x > int(self['width']) -portee:
            self.decalageX(-sensi)
    def decalageX(self, x):
        self.dX += ZOOM*x
    def decalageY(self, y):
        self.dY += ZOOM*y
             
    
class CanvasElement:
    liste_element = {}
    def __init__(self, canvas, type_dessin, coords:list, **kwargs):
        self.canvas = canvas
        self.type_dessin = type_dessin
        self.coords = coords
        self.id = None
        self.kwargs = kwargs
        if canvas in CanvasElement.liste_element.keys():
            CanvasElement.liste_element[canvas].append(self)
        else:
            CanvasElement.liste_element[canvas] = [self]
        self.decalageX = 0 
        self.decalageY = 0
    def get_liste(self,canvas):
        return CanvasElement.liste_element[canvas]
    def update(self):
        if self.id == None:
            self.init_dessin()
        else:
            coords = self.coords.copy()
            x,y = self.decalageX, self.decalageY
            coords[1] += y
            i=1
            while 2*i+1<len(coords):
                coords[2*i+1] += y
                i += 1
            coords[0] += x
            i=1
            while 2*i<len(coords):
                coords[2*i] += x
                i += 1
            self.canvas.coords(self.id,*coords)
            self.canvas.itemconfig(self.id,**self.kwargs)
    def update_coords(self,new_coords:list):
        self.coords = new_coords
    def init_dessin(self):
        if self.type_dessin == 'line':
            self.id = self.canvas.create_line(*self.coords, **self.kwargs)
        elif self.type_dessin == 'oval':
            self.id = self.canvas.create_oval(*self.coords,**self.kwargs)
        elif self.type_dessin == 'text':
            self.coords = [self.coords[0],self.coords[1]]
            self.id = self.canvas.create_text(*self.coords,**self.kwargs)
        else:
            self.id = eval('self.canvas.create_{0}'.format(self.type_dessin))(*self.coords,**self.kwargs)

data0 = None
class PanedWindowData(PanedWindow):
    COULEUR = 'cyan'
    type_pnd = 'DATA'
    def __init__(self, master, numero):
        self.numero = numero
        self.master = master
        PanedWindow.__init__(self,master, bg = PanedWindowData.COULEUR,orient = VERTICAL)
        self.add(Label(master,text= "data{0} :".format(numero) , bg = PanedWindowData.COULEUR))
        self.add(Label(master, bg = PanedWindowData.COULEUR,text="Nom du fichier contenant les données : "))
        
        self.entree_nom = Entry(master)
        self.entree_nom.insert(0,"data/")

        self.add(self.entree_nom)
        self.add(Label(master,text="Séparation :", anchor="w", bg = PanedWindowData.COULEUR))
        self.entree_separate = Entry(master)
        self.entree_separate.insert(0,',')
        self.add(self.entree_separate)
        self.add(Button(self,text="Ajouter les données sélectionnées à data{0} ".format(numero), command = self.get_data))
        self.text_area = scrolledtext.ScrolledText(self.master, bg = PanedWindowData.COULEUR, wrap=WORD, width=40, height=10)
        self.add(Button(self,text="Mélanger", command=self.shuffle))
        self.add(Button(self,text="Effacer les données", command = self.delete_data))
    def delete_data(self):
        self.text_area.delete("1.0",END)
    def shuffle(self):
        global data0, data1, data2 
        if self.numero == 0:
            data0 = data0[np.random.permutation(data0.shape[0])]
            data = data0
        if self.numero == 1:
            data1 = data1[np.random.permutation(data1.shape[0])]
            data = data1
        self.delete_data()
        self.text_area.insert(INSERT,np.array2string(data, threshold=np.inf, separator=', '))
    def get_data(self):
        global data0, data1, data2
        try:
            with open(self.entree_nom.get(),'r') as f:
                file = f.read()
        except Exception as e:
            alerte(e)
            return None
        tab = []
        for i in (file.split('\n')):
            if i != '':
                tab.append(i.split(self.entree_separate.get()))
        data = np.array(tab, dtype=object)
        
        N,P = data.shape
        for i in range(N):
            for j in range(P):
                try:
                    data[i,j] = float(data[i,j])
                except:
                    pass
        
        if self.numero == 0:
            data0 = data
            data = data0
        if self.numero == 1:
            data1 = data
            data = data1
        if self.numero == 2:
            data2 = data
            data = data2
        self.delete_data()
        self.text_area.insert(INSERT,np.array2string(data, threshold=np.inf, separator=', '))

entr0 = datavide()
val0 = datavide()
test0 = datavide()
class PanedWindowDivision(PanedWindow):
    COULEUR = 'red'
    type_pnd = 'DIVISION'
    def __init__(self,master, numero):
        self.master = master
        self.numero = numero
        PanedWindow.__init__(self,master,bg=PanedWindowDivision.COULEUR, orient=VERTICAL)
        self.add(Label(master,bg = PanedWindowDivision.COULEUR,  text = 'decoupage{0}'.format(numero)))
        self.add(Label(master,bg = PanedWindowDivision.COULEUR,text="Données d'entraînement (entr{0}):".format(numero),anchor='w'))
        self.entree_application = Entry(self)
        self.entree_application.insert(0,"data0[1:N//2,:]")
        self.add(self.entree_application)
        self.add(Label(master,bg = PanedWindowDivision.COULEUR,text="Données de validation (val{0}):".format(numero), anchor="w"))
        self.entree_validation = Entry(self)
        self.entree_validation.insert(0,"data0[N//2:3*N//4,:]")
        self.add(self.entree_validation)
        self.add(Label(master,bg = PanedWindowDivision.COULEUR,text="Données de test (test{0}):".format(numero),anchor='w'))
        self.entree_test = Entry(self)
        self.entree_test.insert(0,"data0[3*N//4:N,:]")
        self.add(self.entree_test)
        self.add(Button(self,text="Appliquer ce découpage", command = self.decouper))
        w,h = 200,50
        self.canvas = Canvas(self, width = w,height = h)
        self.canvas.create_rectangle(5,5,w-5,h-5, fill='grey')
        self.add(self.canvas)
    def decouper(self):
        global data0,entr0,val0,test0
        if self.numero == 0:
            N,P = data0.shape
            entr0 = eval(self.entree_application.get())
            val0 = eval(self.entree_validation.get())
            test0 = eval(self.entree_test.get())
            entr0 = entr0.copy()
            val0 = val0.copy()
            test0 = test0.copy()
            a,b,c = entr0.shape[0], val0.shape[0], test0.shape[0]
            if a+b+c == 0:
                alerte("Pas de données sélectionnées")
            w,h = 200,50
            l1 = int(a/(a+b+c)*(w-10))
            l2 = int(b/(a+b+c)*(w-10))
            self.canvas.create_rectangle(5,5,w-5,h-5, fill='blue')
            self.canvas.create_rectangle(5,5,5+l1,h-5,fill = 'red')
            self.canvas.create_rectangle(5+l1,5,5+l1+l2,h-5,fill='green')
            t = 10
            l = [('Apprentissage','red'),('Validation','green'),('Test','blue')]
            for i in range(len(l)):
                text,clr = l[i]
                self.canvas.create_rectangle(5,h+5+2*t*i,5+t,h+5+t+2*t*i,fill=clr)
                self.canvas.create_text(5+2*t, h+5+t//2+2*t*i, text=text,anchor="w")
def alerte(message):
    messagebox.showerror("Erreur",str(message))

I = Interface()

