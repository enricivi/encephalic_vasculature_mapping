import argparse
import numpy as np
import progressbar
from xml.etree.ElementTree import parse, SubElement, tostring
from sklearn.cluster import mean_shift


def marker_position(root_pos):
    marker = "##x,y,z,radius,shape,name,comment,color_r,color_g,color_b" + "\n"
    for vertex in root_pos.iterfind("node"):
        x = vertex.find('X').text.replace(' ','').replace('\n','')
        y = vertex.find('Y').text.replace(' ','').replace('\n','')
        z = vertex.find('Z').text.replace(' ','').replace('\n','')
        marker += x + "," + y + "," + z + ",20,1,"
        marker += vertex.attrib["id"] + ",_,255,0,0" + "\n"
    return marker

def reassemble_positions(root_pos, cluster, cluster_pos):
    cluster_id = cluster[0].attrib["id"]
    for c in cluster:
        root_pos.remove(c)
    """
    for c in cluster:
        for vertex in root_pos.findall("node"):
            if vertex.attrib["id"] == c.attrib["id"]:
                root_pos.remove(vertex)
                break
    """
    node = SubElement(root_pos, 'node', id=cluster_id)
    SubElement(node, 'X').text = str(int(cluster_pos[0]))
    SubElement(node, 'Y').text = str(int(cluster_pos[1]))
    SubElement(node, 'Z').text = str(int(cluster_pos[2]))

def reassemble_adjacent(root_adj, cluster, bandwidth):
    cluster_id = cluster[0].attrib["id"]
    id_elements_in_cluster = [c.attrib["id"] for c in cluster]
    neighbor_elements_in_cluster = []
    for vertex in root_adj.findall("node"):
        if vertex.attrib["id"] in id_elements_in_cluster:
            neighbor_elements_in_cluster.extend(vertex.findall("node_neighbour"))
            root_adj.remove(vertex)
        else:
            for neighbor in vertex.iterfind("node_neighbour"):
                if neighbor.attrib["id_neighbor"] in id_elements_in_cluster:
                    neighbor.attrib["id_neighbor"] = cluster_id
    tmp = []
    for n in neighbor_elements_in_cluster:
        if not(n.attrib["id_neighbor"] in id_elements_in_cluster):
            tmp.append((n.attrib["id_neighbor"], float(n.text)))
        elif float(n.text) >= 1.5*bandwidth:
            tmp.append((cluster_id, float(n.text)))
    neighbor_elements_in_cluster = tmp
    """
    neighbor_elements_in_cluster = [(n.attrib["id_neighbor"], float(n.text))
                                    for n in neighbor_elements_in_cluster
                                    if not(n.attrib["id_neighbor"] in id_elements_in_cluster)]
    """
    node = SubElement(root_adj, 'node', id=cluster_id)
    for neighbor in neighbor_elements_in_cluster:
        SubElement(node, 'node_neighbour', id_neighbor=neighbor[0]).text = str(neighbor[1])

def main(adjacent, position, bandwidth, output):
    position = parse(position).getroot()
    vertex = []
    vertex_id = []
    for node in position.iterfind("node"):
        vertex.append([int(node.find('X').text),
                       int(node.find('Y').text),
                       int(node.find('Z').text)])
        vertex_id.append(node.attrib["id"])
    vertex = np.asarray(vertex)
    vertex_id = np.asarray(vertex_id)
    print "total nodes: " + str(len(vertex))
    print "bandwidth: " + str(bandwidth)
    cluster_centers, labels = mean_shift(vertex, bandwidth) #lables = cluster labels for each point
    print "number of cluster: " + str(cluster_centers.shape[0])
    adjacent = parse(adjacent).getroot()
    set_of_labels = set(labels)
    widgets = ["reassemble graph: ", progressbar.Percentage(), ' ',
               progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(set_of_labels))
    pbar.start()
    position_findall_node = position.findall("node")
    i = 0 #only for progress bar
    for label in set_of_labels:
        pbar.update(i)
        i += 1  #only for progress bar
        index = np.where(labels == label)[0]
        nodes = [node for node in position_findall_node if node.attrib["id"] in vertex_id[index]]
        reassemble_positions(position, nodes, cluster_centers[label])
        reassemble_adjacent(adjacent, nodes, bandwidth)
    pbar.finish()
    print "save output file: "
    print "    > position_list.xml"
    with open(output + "position_list.xml", "wb") as position_list:
        position_list.write(tostring(position))
    print "    > adjacent_list.xml"
    with open(output + "adjacent_list.xml", "wb") as adjacent_list:
        adjacent_list.write(tostring(adjacent))
    print "    > position.marker"
    with open(output + "position.marker", "wb") as position_marker:
        position_marker.write(marker_position(position))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("adjacent_list", type=str, help="path to adjacent list (XML)")
    parser.add_argument("position_list", type=str, help="path to position list (XML)")
    parser.add_argument("bandwidth", type=float, help="bandwidth used in the RBF kernel (mean-shift)")
    parser.add_argument("output", type=str, help="path to output folder")
    args = parser.parse_args()
    main(args.adjacent_list, args.position_list, args.bandwidth, args.output)
