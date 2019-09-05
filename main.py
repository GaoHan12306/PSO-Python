from pso_algorithm import *
import csv
import _thread


def main():
	end_edge_cloud_20 = _thread.start_new_thread(end_edge_cloud, (20,))
	end_edge_cloud_40 = _thread.start_new_thread(end_edge_cloud, (40,))
	end_edge_cloud_60 = _thread.start_new_thread(end_edge_cloud, (60,))
	end_edge_cloud_80 = _thread.start_new_thread(end_edge_cloud, (80,))
	end_edge_cloud_100 = _thread.start_new_thread(end_edge_cloud, (100,))

	only_cloud_20 = _thread.start_new_thread(only_cloud, (20,))
	only_cloud_40 = _thread.start_new_thread(only_cloud, (40,))
	only_cloud_60 = _thread.start_new_thread(only_cloud, (60,))
	only_cloud_80 = _thread.start_new_thread(only_cloud, (80,))
	only_cloud_100 = _thread.start_new_thread(only_cloud, (100,))

	end_cloud_20 = _thread.start_new_thread(end_cloud, (20,))
	end_cloud_40 = _thread.start_new_thread(end_cloud, (40,))
	end_cloud_60 = _thread.start_new_thread(end_cloud, (60,))
	end_cloud_80 = _thread.start_new_thread(end_cloud, (80,))
	end_cloud_100 = _thread.start_new_thread(end_cloud, (100,))

	only_end_20 = _thread.start_new_thread(only_end, (20,))
	only_end_40 = _thread.start_new_thread(only_end, (40,))
	only_end_60 = _thread.start_new_thread(only_end, (60,))
	only_end_80 = _thread.start_new_thread(only_end, (80,))
	only_end_100 = _thread.start_new_thread(only_end, (100,))

	input()


def end_edge_cloud(workflow: int):
	'''
	端边云
	:return:
	'''
	ee = EdgeEnvironment('./YOLO1.json')
	pp = ParticleSwarm(ee)
	result, fitness, time, energy = pp.start_pso()
	with open('./YOLO1_' + str(workflow) + '.csv', 'a', newline='') as f:
		# newline防止writerows自动换行
		f_csv = csv.writer(f)
		f_csv.writerow(['调度方案'])
		f_csv.writerows(result)
		f_csv.writerow(['适应度值'])
		f_csv.writerow(fitness)
		f_csv.writerow(['时间'])
		f_csv.writerows(time)
		f_csv.writerow(['能耗'])
		f_csv.writerows(energy)


def only_cloud(workflow: int):
	'''
	纯云
	:return:
	'''
	ee = EdgeEnvironment('./YOLO2.json')
	pp = ParticleSwarm(ee)
	result, fitness, time, energy = pp.start_pso()
	with open('./YOLO2_' + str(workflow) + '.csv', 'a', newline='') as f:
		# newline防止writerows自动换行
		f_csv = csv.writer(f)
		f_csv.writerow(['调度方案'])
		f_csv.writerows(result)
		f_csv.writerow(['适应度值'])
		f_csv.writerow(fitness)
		f_csv.writerow(['时间'])
		f_csv.writerows(time)
		f_csv.writerow(['能耗'])
		f_csv.writerows(energy)


def end_cloud(workflow: int):
	'''
	端云
	:return:
	'''
	ee = EdgeEnvironment('./YOLO4.json')
	pp = ParticleSwarm(ee)
	result, fitness, time, energy = pp.start_pso()
	with open('./YOLO4_' + str(workflow) + '.csv', 'a', newline='') as f:
		# newline防止writerows自动换行
		f_csv = csv.writer(f)
		f_csv.writerow(['调度方案'])
		f_csv.writerows(result)
		f_csv.writerow(['适应度值'])
		f_csv.writerow(fitness)
		f_csv.writerow(['时间'])
		f_csv.writerows(time)
		f_csv.writerow(['能耗'])
		f_csv.writerows(energy)


def only_end(workflow: int):
	'''
	纯端
	:return:
	'''
	ee = EdgeEnvironment('./YOLO7.json')
	pp = ParticleSwarm(ee)
	result, fitness, time, energy = pp.start_pso()
	with open('./YOLO7_' + str(workflow) + '.csv', 'a', newline='') as f:
		# newline防止writerows自动换行
		f_csv = csv.writer(f)
		f_csv.writerow(['调度方案'])
		f_csv.writerows(result)
		f_csv.writerow(['适应度值'])
		f_csv.writerow(fitness)
		f_csv.writerow(['时间'])
		f_csv.writerows(time)
		f_csv.writerow(['能耗'])
		f_csv.writerows(energy)


if __name__ == '__main__':
	main()

