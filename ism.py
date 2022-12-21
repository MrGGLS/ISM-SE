import numpy as np
import pandas as pd
import networkx as nx
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['HeiTi', 'KaiTi', 'SimHei', 'FangSong']
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='data.csv',help='data file path')
parser.add_argument('-t', '--type', default='list', help='data type: matrix or list')
parser.add_argument('-o', '--output', default='output.csv', help='multigraph file path')
args = parser.parse_args()

# nodes = 'S1 S2 S3 S4 S5 S6 S7'.split(' ')
nodes = 'CPU 显卡 内存 硬盘 需求 预算 售后 性能 续航 外观 重量 材质 时尚'.split(' ')

def list_to_matrix(adj_list, n_nodes):
  adj_mat = np.zeros((n_nodes, n_nodes), dtype=np.bool8)
  for i, j in adj_list:
    adj_mat[i-1][j-1] = 1
  return adj_mat

def read_data(data_path, data_type='list'):
  data = pd.read_csv(data_path, sep=',', header=None)
  data = data.to_numpy()
  n_nodes = data.max()
  if data_type == 'list':
    data = list_to_matrix(data, n_nodes)
  return data, n_nodes


def reduce_matrix(M):
  M = np.array(M, dtype=np.bool8)
  n = len(M[0])
  strong_node_pairs = {}
  for i in range(n):
    f = False
    for k in strong_node_pairs:
      if i + 1 in strong_node_pairs[k]:
        f = True
        break
    if f:
      continue
    for j in range(i+1, n):
      if np.all(M[i] == M[j]) and np.all(M[:][i] == M[:][j]):
        p_i, p_j = i, j
        if p_i + 1 in strong_node_pairs:
          strong_node_pairs[p_i+1].append(p_j+1)
        else:
          strong_node_pairs[p_i+1] = [p_j+1]
  exclude_set = set()
  for k in strong_node_pairs:
    exclude_set = exclude_set.union(set(strong_node_pairs[k]))
  exclude_set = list(exclude_set)
  new_mat = []
  for i in range(n):
    if i + 1 in exclude_set:
      continue
    for j in range(n):
      if j + 1 in exclude_set:
        continue
      new_mat.append(M[i][j])
  new_mat = np.array(new_mat, dtype=np.bool8).reshape(n-len(exclude_set), -1)
  _nodes = []
  for i in range(len(nodes)):
    if i + 1 in exclude_set:
      continue
    if i + 1 in strong_node_pairs:
      _nodes.append(nodes[i])
      for node in strong_node_pairs[i+1]:
        _nodes[-1] += ' + ' + nodes[node-1]
    else:
      _nodes.append(nodes[i])
  # print(f'exclude set {exclude_set} \n _nodes {_nodes}')
  return new_mat, strong_node_pairs, exclude_set, _nodes


def backbone_matrix(M):
  n = len(M[0])
  I = np.identity(n) > 0.5
  M = M ^ np.dot(M^I, M^I) ^ I
  for i in range(n):
    for j in range(n):
      if M[i][j] and i != j:
        for k in range(n):
          if M[i][k] and M[k][j]:
            M[i][j] = False
  return M


def save_result(M, node_pairs, exclude_set, n_nodes):
  index = [i+1 for i in range(n_nodes)]
  index = set(index).difference(set(exclude_set))
  index = sorted(list(index))
  G = nx.DiGraph()
  edges = []
  for i in range(len(index)):
    for j in range(len(index)):
      if M[i][j]:
        edges.append((nodes[index[i]-1], nodes[index[j]-1]))
  for k in node_pairs:
    tmp = k
    for node in node_pairs[k]:
      edges.append((nodes[tmp-1], nodes[node-1]))
      edges.append((nodes[node-1], nodes[tmp-1]))
      tmp = node
  G.add_edges_from(edges)
  # print(edges)
  nx.draw_shell(G, with_labels=True)
  # plt.show()
  plt.savefig(args.output)


def idx_to_node(idx_set):
  new_set = []
  for s in idx_set:
    new_set.append([nodes[idx-1] for idx in s])
  return new_set


def ISM():
  A, n_nodes = read_data(args.data, args.type)
  print('=' * 10 + 'original matrix A' + '=' * 10)
  print(tabulate(pd.DataFrame(A+0, index=nodes, columns=nodes), headers='keys', tablefmt='psql'))
  pd.DataFrame(A+0, index=nodes, columns=nodes).to_csv('./original matrix.csv')

  I = np.identity(n_nodes) > 0.5
  M = A + I
  print('=' * 10 + 'adjacent matrix I' + '=' * 10)
  print(tabulate(pd.DataFrame(M+0, index=nodes, columns=nodes), headers='keys', tablefmt='psql'))
  pd.DataFrame(M+0, index=nodes, columns=nodes).to_csv('./adjacent matrix.csv')

  _M = M
  r = 0
  while True:
    M = M @ (A+I)
    r += 1
    if np.sum(M != _M) == 0:
      break
    _M = M
  print('=' * 10 + 'reachable matrix M' + '=' * 10)
  print(f'r = {r}')
  print(tabulate(pd.DataFrame(M+0, index=nodes, columns=nodes), headers='keys', tablefmt='psql'))
  pd.DataFrame(M+0, index=nodes, columns=nodes).to_csv('./reachable matrix.csv')

  r_set, a_set, c_set, b_set = [], [], [], []
  tmp_set = []
  for i in range(n_nodes):
    for j in range(n_nodes):
      if M[i][j]:
        tmp_set.append(j+1)
    r_set.append(sorted(tmp_set.copy()))
    tmp_set.clear()

  for i in range(n_nodes):
    for j in range(n_nodes):
      if M[j][i]:
        tmp_set.append(j+1)
    a_set.append(sorted(tmp_set.copy()))
    tmp_set.clear()

  for i in range(n_nodes):
    if len(r_set[i]) == 0 or len(a_set[i]) == 0:
      c_set.append([])
    c_set.append(sorted(list(set(r_set[i])&set(a_set[i]))))

  for i in range(n_nodes):
    b_set.append(sorted(list([] if a_set[i] != c_set[i] else list(a_set[i]))))
  
  # print(f'r_set: {idx_to_node(r_set)}')
  # print(f'a_set: {idx_to_node(a_set)}')
  # print(f'c_set: {idx_to_node(c_set)}')
  # print(f'b_set: {idx_to_node(b_set)}')
  
  level_set = []
  cnt = 0
  while cnt < len(r_set):
    for i in range(len(r_set)):
      if r_set[i] == c_set[i]:
        for elem in r_set[i]:
          tmp_set.append(elem)
    for j in range(len(r_set)):
      for node in tmp_set:
        if node in r_set[j]:
          r_set[j].remove(node)
          if len(r_set[j]) == 0:
            cnt += 1
        if node in c_set[j]:
          c_set[j].remove(node)
    level_set.append(list(set(tmp_set.copy())))
    tmp_set.clear()

  print(f'level_set: {idx_to_node(level_set)}')    

  reduced_mat, strong_node_pairs, exclude_set, _nodes = reduce_matrix(M)
  print('=' * 10 + 'reduced matrix M\'' + '=' * 10)       
  print(tabulate(pd.DataFrame(reduced_mat+0, index=_nodes, columns=_nodes), headers='keys', tablefmt='psql'))
  pd.DataFrame(reduced_mat+0, index=_nodes, columns=_nodes).to_csv('./reduced matrix.csv')

  print(f'strong_node_pairs: {strong_node_pairs}')

  backbone_mat = backbone_matrix(reduced_mat)
  print('=' * 10 + 'backbone matrix M\'' + '=' * 10)       
  print(tabulate(pd.DataFrame(backbone_mat+0, index=_nodes, columns=_nodes), headers='keys', tablefmt='psql'))
  pd.DataFrame(backbone_mat+0, index=_nodes, columns=_nodes).to_csv('./backbone matrix.csv')

  save_result(backbone_mat, strong_node_pairs, exclude_set, n_nodes)


if __name__ == '__main__':
  ISM()