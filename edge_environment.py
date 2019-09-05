import json
import numpy as np

np.set_printoptions(suppress=True)


class EdgeEnvironment:
	'''
	部署边缘计算环境，初始化工作流、任务、边缘设备等
	'''

	def __init__(self, path: str, workflow_num=100):
		'''
		初始化边缘计算环境
		:param path: json文件路径
		:param workflow_num: 总工作流数目
		'''
		load_f = open(path, mode='r')
		load_dict = json.load(load_f)

		self.workflow_num = workflow_num
		(
			self.task_num,
			self.workflow_task_workload,
			self.workflow_task_inputSize,
			self.workflow_task_outputSize,
			self.task_prior,
			self.task_next
		) = self.load_workflow_task(load_dict)
		(
			self.device_num,
			self.end_device,
			self.device_type,
			self.device_procRate
		) = self.load_device(load_dict)
		self.workflow_task_deadline = self.predict_deadline()

	def load_workflow_task(self, load_dict):
		'''
		初始化工作流任务的负载、输入数据大小、输出数据大小、前驱和后继
		:param path:json文件路径
		:return:工作流任务数、工作流任务的负载、输入数据大小、输出数据大小、前驱和后继
		'''
		task_num = len(load_dict["tasks"])

		workload = np.zeros(
			shape=(self.workflow_num, task_num),
			dtype=np.float
		)  # 工作流任务负载

		inputSize = np.zeros(
			shape=(self.workflow_num, task_num),
			dtype=np.float
		)  # 工作流任务输入数据大小

		outputSize = np.zeros(
			shape=(self.workflow_num, task_num),
			dtype=np.float
		)  # 工作流任务输出数据大小

		# 初始化
		for i in range(self.workflow_num):
			for j in range(task_num):
				workload[i, j] = load_dict['tasks'][j]['workload'] * (i + 1)
				inputSize[i, j] = load_dict['tasks'][j]['inputSize'] * (i + 1)
				outputSize[i, j] = load_dict['tasks'][j]['inputSize'] * (i + 1)

		# 前驱后继
		prior = []
		next = []

		# 初始化
		for i in range(task_num):
			prior.append(load_dict['tasks'][i]['prior'])
			next.append(load_dict['tasks'][i]['next'])

		return (
			task_num,
			workload,
			inputSize,
			outputSize,
			prior,
			next
		)

	def load_device(self, load_dict):
		'''
		初始化终端设备和边缘环境中各个设备的信息
		:param path: json文件的路径
		:return: 关于终端设备信息的字典和关于边缘环境信息的列表
		'''
		end_device = load_dict['end']
		devices = load_dict['devices']
		device_type = []
		device_procRate = []

		for i in range(len(devices)):
			device_type.append(devices[i]['type'])
			device_procRate.append(devices[i]['procRate'])

		return (
			len(devices),
			end_device,
			device_type,
			np.array(device_procRate)
		)

	def predict_deadline(self):
		'''
		根据边缘计算设备的运算能力设定工作流任务的时间阈值
		:return:每个工作流的每个任务的执行时间阈值
		'''
		deadline = self.workflow_task_workload / np.mean(self.device_procRate)
		return deadline
