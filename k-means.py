#!/usr/bin/env python
# -*- coding: utf-8 -*-
# K-means (Duda & Hart) #
# sampleSpace -> Conjunto de puntos a agrupar
# classes -> Número de clases para la agrupación
# clusters -> Conjunto de clusters donde acumular los refinamientos 
import matplotlib.pyplot as plt
from random import randint
from sys import maxint

plot_colours = ['go','co','mo','yo','ko','bo','ro']

# Distribución cíclica de muestras entre clusters (probar con otras inicializaciones a ver si converge más rápido con otras) #
def generate_random_clusters(clusters,classes,sampleSpace):
	for n in xrange(len(sampleSpace)):
		clusters[n%len(classes)].append(sampleSpace[n])
	return clusters

# K-Means #
def kmeans(clusters,classes,sampleSpace,max_iter,verbose):
	representatives = [[] for n in xrange(len(classes))]
	J,count_iter,transfers = 0,0,False
	for c in xrange(len(classes)):
		sumx = [0,0]
		for x in clusters[c]: sumx = vectorsum(x,sumx)
		representatives[c] = vectorxconstant(sumx,(1.0/len(clusters[c])))
	if verbose : print "Representantes: " + str(representatives)
	while(not transfers and count_iter < max_iter):
		count_iter,transfers = (count_iter+1),False
		# Por cada muestra i del espacio muestral #
		for sample in sampleSpace:
			# Conocer cluster de i (Prefijo i -> indice) #
			iclusteri = get_sample_cluster(clusters,sample)
			if len(clusters[iclusteri]) > 1:
				icluster_optimum = None; auxMin = maxint; aux = None; VJ = 0
				# Calcular ( argmin\_j!=i nj / (nj+1) * ||x-mj||^2 / ) que minimize el SEC de esa muestra #
				for iclusterj in xrange(len(clusters)):
					if iclusterj != iclusteri:
						aux = vectorpow2(vectorabs(vectordifference(sample,representatives[iclusterj])))*(len(clusters[iclusterj])/len(clusters[iclusterj])+1.0)
						if(aux<auxMin):
							auxMin = aux
							icluster_optimum = iclusterj
				if verbose : print "Sample " + str(sample) + " mas cerca de " + str(representatives[icluster_optimum]) + " con SEC " + str(auxMin)
				# Calcular gradiente de J #
				VJ = vectorpow2(vectorabs(vectordifference(sample,representatives[icluster_optimum])))*(len(clusters[icluster_optimum])/len(clusters[icluster_optimum])+1.0) - float((vectorpow2(vectorabs(vectordifference(sample,representatives[iclusteri])))*(len(clusters[iclusteri]))/len(clusters[iclusteri])-1.0))
				# Si el SEC se reduce cambiando la muestra de un cluster a otro, cambiar #
				if(VJ<0):
					transfers = True
					if verbose : print "Sample movida del cluster: " + str(iclusteri) + " al cluster: " + str(icluster_optimum)
					# Recalcular representantes #
					representatives[iclusteri]        = vectordifference(representatives[iclusteri],(vectordconstant(vectordifference(sample,representatives[iclusteri]),(len(clusters[iclusteri])-1.0))))
					representatives[icluster_optimum] = vectorsum(representatives[icluster_optimum],(vectordconstant(vectordifference(sample,representatives[icluster_optimum]),(len(clusters[icluster_optimum])-1.0))))
					# Transferencia de la muestra #
					clusters[iclusteri].remove(sample)
					clusters[icluster_optimum].append(sample)
					J += VJ
	return [clusters,representatives,J]
	
## Funciones vectores ##
def vectorxvector(v,y):
	s = 0
	for n in xrange(len(v)):
		s += v[n] * y[n]
	return s
	
def vectordconstant(v,a):
	return [x/a for x in v]
	
def vectorxconstant(v,a):
	return [x*a for x in v]

def vectordifference(v,y):
	return [v[n]-y[n] for n in xrange(len(v))]

def vectorsum(v,y):
	return [v[n]+y[n] for n in xrange(len(v))]	

def vectorabs(v):
	return [abs(i) for i in v]	

def vectorpow2(v):
	return vectorxvector(v,v)

# Funciones k-means #
def get_sample_cluster(clusters,sample):
	for icluster in xrange(len(clusters)):
		for point in clusters[icluster]:
			if point == sample: return icluster
	return -1

def genera_espacio_muestral_aleatorio(n,inf,sup):
	return [[randint(inf,sup),randint(inf,sup)] for x in xrange(n)]

# Funciones Matplotlib #
def generar_colores(numclasses):
	return [plot_colours.pop(randint(0,len(plot_colours)-1)) for x in xrange(numclasses)]
	
## EntryPoint ##	
if __name__ == '__main__':
	# Test #
	sampleSpace = [[1.2,2.3],[7,5],[1.7,3.5],[7,10],[7,7],[1,4],[1,5],[10,1],[1,6],[10,2],[10,4],[10,6],
				   [5,6.2],[3.5,1.7],[6.3,7.9],[10.10,12.3],[1.3,5.6],[2.7,1.8],[13.13,13.16],[8.5,9.6],
				   [1.3,4.2],[7.2,7.9]]# || genera_espacio_muestral_aleatorio(30,1,30)
	classes     = ["clase1","clase2"]
	clusters    = [[],[]]
	clusters    = generate_random_clusters(clusters,classes,sampleSpace)
	ret = kmeans(clusters,classes,sampleSpace,10000,True) 
	## Paint with matplotlib ##
	plot_colours = generar_colores(len(classes))
	for cluster in ret[0]:
		x = []
		y = []
		ax = plt.gca()
		ax.set_xlabel('Dimension 1')
		ax.set_ylabel('Dimension 2')
		ax.set_title('K-means ' + str(len(classes)) + ' classes')	
		ax.legend([i for i in classes])
		for point in cluster:
			x.append(point[0]); y.append(point[1])
		plt.plot(x,y,plot_colours.pop())
	plt.show()
