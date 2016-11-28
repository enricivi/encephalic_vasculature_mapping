# @String img_file
# @String output

import ij.process as process
from ij import IJ, ImagePlus
from sc.fiji.skeletonize3D import Skeletonize3D_
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_, Point
from math import fabs
from xml.etree import ElementTree
from xml.dom import minidom

def markerPosition(adjacent_list):
	marker= "##x,y,z,radius,shape,name,comment,color_r,color_g,color_b" + "\n"
	for l in adjacent_list:
		position= l[0].getPoints()[0]
		marker= marker + str(position.x) + "," + str(position.y) + "," + str(position.z) 
		marker= marker + "," + "20,1," + str(l[0]).split('@', 1)[1] + ",_,255,0,0" + "\n"
	return marker

def createXMLlocation(adjacent_list, w, h, z):
	xml= ElementTree.Element('position_list')
	##
	node= ElementTree.SubElement(ElementTree.SubElement(xml, 'info_limits'), 'limit_ends')
	ElementTree.SubElement(node, 'X').text = str(w-1)
	ElementTree.SubElement(node, 'Y').text = str(h-1)
	ElementTree.SubElement(node, 'Z').text = str(z-1)
	##
	for l in adjacent_list:
		position= l[0].getPoints()[0]
		node= ElementTree.SubElement(xml, 'node', id=str(l[0]).split('@', 1)[1])
		ElementTree.SubElement(node, 'X').text = str(position.x)
		ElementTree.SubElement(node, 'Y').text = str(position.y)
		ElementTree.SubElement(node, 'Z').text = str(position.z)
	try:
		xml= ElementTree.tostring(xml, 'utf-8')
	except: xml= "<Error />"
	return minidom.parseString(xml).toprettyxml(indent="    ")

def createXMLadjacent(adjacent_list):
	xml= ElementTree.Element('adjacent_list')
	for l in adjacent_list:
		node= ElementTree.SubElement(xml, 'node', id=str(l[0]).split('@', 1)[1])
		for n in l[1]:
			neighbours= ElementTree.SubElement(node, 'node_neighbour', id_neighbor=str(n[0]).split('@', 1)[1])
			neighbours.text= str(n[1])
	try:
		xml= ElementTree.tostring(xml, 'utf-8')
	except: xml= "<Error />"
	return minidom.parseString(xml).toprettyxml(indent="    ")

def neighborhood(edge, v):
	n= []
	for e in edge:
		n.append( (e.getOppositeVertex(v), e.getLength()) )
	return n

def main():
	print("open image...")
	skel= IJ.openImage(img_file)
	height= skel.getHeight()
	width= skel.getWidth()
	nslices= skel.getNSlices()

	print "skeletonizing image..."
	IJ.run(skel, "Skeletonize (2D/3D)", "")
	#skel.show()

	print("analyzing skeleton...")
	analyzeSkel= AnalyzeSkeleton_()
	analyzeSkel.calculateShortestPath= False
	analyzeSkel.setup('', skel)
	result= analyzeSkel.run(AnalyzeSkeleton_.NONE, False, False, None, True, False)

	print("saving result...")
	graph= result.getGraph()

	adjacent_list= []
	for j in xrange(len(graph)):
		g= graph[j]
		for v in g.getVertices():
			v.setVisited(False)
		node= [ g.getVertices()[0] ]
		while ( len(node) != 0 ):
			n= node.pop(0)
			n.setVisited(True)
			node_neighbours= neighborhood(n.getBranches(), n)
			adjacent_list.append( (n, node_neighbours) )
			for c in node_neighbours:
				if ( (not(c[0].isVisited())) and (not(c[0] in node)) ):
					node.append( c[0] )
	try:
		outputfile= open(output + "adjacent_list.xml","wb")
		outputfile.write(createXMLadjacent(adjacent_list))
		outputfile= open(output + "position_list.xml","wb")
		outputfile.write(createXMLlocation(adjacent_list, width, height, nslices))
		outputfile= open(output + "position.marker", "wb")
		outputfile.write(markerPosition(adjacent_list))
	except: 
		print("impossible to create or write files")
	finally: 
		outputfile.close()

if __name__ == "__main__":
	main()
