
import tensorflow as tf

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
pb_path = "./DH_model.pb"
graph = load_pb(pb_path)
input = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('output:0')
print("Success")
# sess.run(output, feed_dict={input: some_data})



# pb_path = './frozen_model.pb'
# with tf.compat.v1.Session() as sess:
#    print("load graph")
#    with tf.io.gfile.GFile(pb_path,'rb') as f:
#        graph_def = tf.compat.v1.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf.import_graph_def(graph_def, name='')
#    graph_nodes=[n for n in graph_def.node]
#    names = []
#    for t in graph_nodes:
#       names.append(t.name)
#    print(names)
   