(.venv) sqp17@sun:~/path/to/project-root/third_party/tgl$ uv run python check_if_reverse_added.py --data REDDIT
dataset: REDDIT
graph: DATA/REDDIT/full_graph_with_reverse_edges.npz
graph_type: full_graph_with_reverse_edges
eligible_edges: 672447
graph_entries: 1344894
reverse_added: inconsistent
detail: graph does not fully match either generation mode.
missing_forward_edges: 27512
  eid=19, src=20, dst=10015, time=84.62100000000001
  eid=32, src=33, dst=10023, time=177.56099999999998
  eid=73, src=73, dst=10048, time=346.23900000000003
  eid=75, src=75, dst=10017, time=352.37800000000004
  eid=114, src=103, dst=10063, time=520.9209999999999
missing_reverse_edges: 27512
  eid=19, src=20, dst=10015, time=84.62100000000001
  eid=32, src=33, dst=10023, time=177.56099999999998
  eid=73, src=73, dst=10048, time=346.23900000000003
  eid=75, src=75, dst=10017, time=352.37800000000004
  eid=114, src=103, dst=10063, time=520.9209999999999

We regenerated, now:
(.venv) sqp17@sun:~/path/to/project-root/third_party/tgl$ uv run python check_if_reverse_added.py --data REDDIT
dataset: REDDIT
graph: DATA/REDDIT/full_graph_with_reverse_edges.npz
graph_type: full_graph_with_reverse_edges
eligible_edges: 672447
graph_entries: 1344894
reverse_added: yes
detail: every eligible edge appears once in the forward direction and once in the reverse direction with the same eid/time.
