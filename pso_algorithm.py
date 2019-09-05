from edge_environment import *

# 常量
c1: float = 2
c2: float = 2
r1: float = 0.5
r2: float = 0.5
w: float = 1.1
LAN: float = 12.5  # 100Mbps局域网
WAN: float = 1.25  # 10Mbps广域网


class ParticleSwarm:
	'''
	粒子群核心算法模块
	'''

	def __init__(self, env: EdgeEnvironment, iter=200, pNum=30):
		'''
		初始化粒子群
		:param env: 绑定的边缘计算环境
		:param iter: 粒子群算法迭代次数
		:param pNum: 粒子数
		'''
		self.env = env  # 边缘环境
		self.iter_num = iter  # 迭代次数
		self.particle_num = pNum  # 粒子数
		self.particle_swarm = np.random.randint(  # 粒子群
			low=0,
			high=self.env.device_num,
			size=self.env.workflow_num * self.env.task_num * self.particle_num
		).reshape((
			self.env.workflow_num,
			self.particle_num,
			self.env.task_num
		))
		self.velocity = np.random.randint(  # 粒子速度
			low=-self.env.device_num,
			high=self.env.device_num,
			size=self.env.workflow_num * self.env.task_num * self.particle_num
		).reshape((
			self.env.workflow_num,
			self.particle_num,
			self.env.task_num
		))
		self.best_particle = np.copy(self.particle_swarm)  # 粒子最优值
		self.fitness_value = np.full(  # 适应度值
			shape=(self.env.workflow_num, self.particle_num),
			fill_value=np.inf,  # 适应度值初始化为无穷大
			dtype=np.float
		)
		self.global_best_particle = np.full(  # 粒子全局最优值
			shape=(self.env.workflow_num,),
			fill_value=-1,
			dtype=np.int
		)

	def update_velocity(self, workflow: int):
		'''
		更新粒子的速度
		:return:
		'''
		for i in range(self.particle_num):
			rand1 = np.random.random(self.env.task_num)
			rand2 = np.random.random(self.env.task_num)  # 产生两个二维随机数组
			part = w * self.velocity[workflow, i] + c1 * rand1 * (
					self.best_particle[workflow, i] - self.particle_swarm[workflow, i]) \
			       + c2 * rand2 * (self.best_particle[workflow, self.global_best_particle]
			                       - self.particle_swarm[workflow, i])
			self.velocity[workflow, i] = part[workflow].astype(np.int)
			for j in range(self.env.task_num):
				if self.velocity[workflow, i, j] > self.env.device_num:
					self.velocity[workflow, i, j] = 0  # 速度超过了上限则置零

	# print('更新粒子的速度完成')

	def update_position(self, workflow: int):
		'''
		更新粒子的位置
		:return:
		'''
		self.particle_swarm[workflow] = np.mod(self.particle_swarm[workflow] + self.velocity[workflow],
		                                       self.env.device_num)

	# print('更新粒子的位置完成')

	def fitness(self, workflow: int):
		'''
		适应度函数
		:return:
		'''
		time = self.get_total_time(workflow)
		energy = self.get_total_energy(workflow)
		deadline = np.sum(self.env.workflow_task_deadline[workflow])
		fit = np.zeros(self.particle_num, dtype=np.float)

		for i in range(self.particle_num):
			if time[i] < deadline:  # 如果时间超过了deadline
				fit[i] = energy[i]
			else:
				fit[i] = 10 * energy[i] * time[i] / deadline

		return fit  # 返回particle维度的一维数组

	def update_best(self, workflow: int):
		'''
		更新每个粒子的最优值
		:return: None
		'''
		for i in range(self.particle_num):
			if self.fitness(workflow)[i] < self.fitness_value[workflow, i]:
				for j in range(self.env.task_num):
					self.best_particle[workflow, i, j] = self.particle_swarm[workflow, i, j]

		self.fitness_value[workflow] = np.minimum(self.fitness_value[workflow], self.fitness(workflow))

	# print('更新粒子最优值完成')

	def update_global_best(self, workflow: int):
		'''
		更新粒子的全局最优值
		:return: None
		'''
		self.global_best_particle[workflow] = self.fitness_value[workflow].argmin()  # 得到当前工作流全局最优适应度值的下标

	# print('更新粒子全局最优值完成')

	def get_time(self, workflow: int):
		'''
		获取当前工作流的分配模式下每个粒子的每个任务的时间
		:return:三维数组
		'''
		workload = np.broadcast_to(
			array=self.env.workflow_task_workload[workflow],
			shape=(
				self.particle_num,
				self.env.task_num
			)
		)  # 当前工作流的每个任务的负载扩展为二维数组

		procRate = np.zeros_like(workload)
		for i in range(self.particle_num):
			for j in range(self.env.task_num):
				# 根据下标寻找服务器从而获取处理速率
				procRate[i, j] = self.env.device_procRate[self.particle_swarm[workflow, i, j]]

		proc_time = workload / procRate  # 计算执行时间

		# 接受时间和发送时间初始化if
		recv_time = np.zeros_like(self.particle_swarm[workflow])
		recv_time.flags.writeable = True
		send_time = np.zeros_like(self.particle_swarm[workflow])
		send_time.flags.writeable = True

		for i in range(self.particle_num):
			for j in range(self.env.task_num):
				# 获取设备类型，任务前驱、后继等信息
				device = self.particle_swarm[workflow, i, j]
				type = self.env.device_type[device]
				prior = self.env.task_prior[j]
				next = self.env.task_next[j]
				inputSize = self.env.workflow_task_inputSize[workflow, j]
				outputSize = self.env.workflow_task_outputSize[workflow, j]

				# 计算接收时间
				if len(prior) == 0:
					# 这是第一个任务
					if type == 'EndDevice':
						# 第一个任务未卸载（仍在终端执行）
						recv_time[i, j] = 0
					else:
						if type == 'CloudServer':
							# 端云之间为广域网
							recv_time[i, j] = inputSize / WAN
						else:
							recv_time[i, j] = inputSize / LAN
				else:
					if len(prior) == 1:
						# 只有一个前驱，不需要判断去最大值
						if device == self.particle_swarm[workflow, i, prior[0]]:
							# 先判断前面的任务是否与当前任务在同一设备上
							recv_time[i, j] = 0
						else:
							if type == 'CloudServer' or self.env.device_type[
								self.particle_swarm[workflow, i, prior[0]]] == 'CloudServer':
								# 只要有一个设备是云就是广域网传输
								recv_time[i, j] = inputSize / WAN
							else:
								recv_time[i, j] = inputSize / LAN
					else:
						max_recv_time = []
						for x in prior:
							# 当前任务的输入数据大小经由其前驱任务的输出数据大小分别计算
							prior_outputSize = self.env.workflow_task_outputSize[workflow, x]
							if device == self.particle_swarm[workflow, i, x]:
								# 先判断前面的任务是否与当前任务在同一设备上
								max_recv_time.append(0)
							else:
								if type == 'CloudServer' or self.env.device_type[
									self.particle_swarm[workflow, i, x]] == 'CloudServer':
									# 只要有一个设备是云就是广域网传输
									max_recv_time.append(prior_outputSize / WAN)
								else:
									max_recv_time.append(prior_outputSize / LAN)
						recv_time[i, j] = max(max_recv_time)  # 根据关键路径原理，取最大值

				# 计算发送时间
				if len(next) == 0:
					# 最后一个任务执行完成
					if type == 'EndDevice':
						# 最后一个任务已经在终端中
						send_time[i, j] = 0
					else:
						if type == 'CloudServer':
							# 端云之间为广域网
							send_time[i, j] = outputSize / WAN
						else:
							send_time[i, j] = outputSize / LAN
				else:
					if len(next) == 1:
						# 只有一个后继，不需要去判断最大值
						if device == self.particle_swarm[workflow, i, next[0]]:
							send_time[i, j] = 0
						else:
							if type == 'CloudServer' or self.env.device_type[
								self.particle_swarm[workflow, i, next[0]]] == 'CloudServer':
								send_time[i, j] = outputSize / WAN
							else:
								send_time[i, j] = outputSize / LAN
					else:
						max_send_time = []
						for y in next:
							next_inputSize = self.env.workflow_task_inputSize[workflow, y]
							if device == self.particle_swarm[workflow, i, y]:
								max_send_time.append(0)
							else:
								if type == 'CloudServer' or self.env.device_type[
									self.particle_swarm[workflow, i, y]] == 'CloudServer':
									max_send_time.append(next_inputSize / WAN)
								else:
									max_send_time.append(next_inputSize / LAN)
						send_time[i, j] = max(max_send_time)

		return recv_time + proc_time + send_time

	def get_energy(self, workflow: int):
		'''
		获取终端设备的能耗
		:return: 三维数组
		'''
		device_energy = np.zeros_like(self.particle_swarm[workflow], dtype=np.float)
		for i in range(self.particle_num):
			for j in range(self.env.task_num):
				if self.env.task_prior[j] == []:
					# 需要计算发送能耗
					device = self.particle_swarm[workflow, i, j]
					inputSize = self.env.workflow_task_inputSize[workflow, j]
					if self.env.device_type[device] == 'EndDevice':
						device_energy[i, j] += 0
					else:
						if self.env.device_type[device] == 'EdgeServer':
							device_energy[i, j] += inputSize * LAN / self.env.end_device['sendPower']
						else:
							device_energy[i, j] += inputSize * WAN / self.env.end_device['sendPower']

				if self.env.task_next[j] == []:
					# 还需要计算接受能耗
					device = self.particle_swarm[workflow, i, j]
					outputSize = self.env.workflow_task_outputSize[workflow, j]
					if self.env.device_type[device] == 'EndDevice':
						device_energy[i, j] += 0
					else:
						if self.env.device_type[device] == 'EdgeServer':
							device_energy[i, j] += outputSize * LAN / self.env.end_device['recvPower']
						else:
							device_energy[i, j] += outputSize * WAN / self.env.end_device['recvPower']
					continue

				if self.env.device_type[self.particle_swarm[workflow, i, j]] == 'EndDevice':
					# 仅考虑终端设备能耗
					# 获取设备，前驱后继、输入输出数据信息
					device = self.particle_swarm[workflow, i, j]
					prior = self.env.task_prior[j]
					next = self.env.task_next[j]
					inputSize = self.env.workflow_task_inputSize[workflow, j]
					outputSize = self.env.workflow_task_outputSize[workflow, j]
					workload = self.env.workflow_task_workload[workflow, j]
					procRate = self.env.device_procRate[device]

					# 计算执行能耗=时间×功率
					proc_energy = workload / procRate * self.env.end_device['procPower']

					# 计算接收能耗
					recv_energy = 0
					if len(prior) == 0:
						# 这是第一个任务，而且没有卸载
						recv_energy = 0
					else:
						if len(prior) == 1:
							# 只有一个前驱，需要计算接收时间
							if device == self.particle_swarm[workflow, i, prior[0]]:
								# 先判断前面的任务是否与当前任务在同一设备上
								recv_energy = 0
							else:
								if self.env.device_type[self.particle_swarm[workflow, i, prior[0]]] == 'CloudServer':
									recv_energy = inputSize / WAN * self.env.end_device['recvPower']
								else:
									recv_energy = inputSize / LAN * self.env.end_device['recvPower']
						else:
							max_recv_time = []
							for x in prior:
								# 当前任务的输入数据大小经由其前驱任务的输出数据大小分别计算
								prior_outputSize = self.env.workflow_task_outputSize[workflow, x]
								if device == self.particle_swarm[workflow, i, x]:
									# 先判断前面的任务是否与当前任务在同一设备上
									max_recv_time.append(0)
								else:
									if self.env.device_type[self.particle_swarm[i, j, x]] == 'CloudServer':
										max_recv_time.append(prior_outputSize / WAN)
									else:
										max_recv_time.append(prior_outputSize / LAN)
							recv_energy = max(max_recv_time) * self.env.end_device['recvPower']

					# 计算发送能耗
					send_energy = 0
					if len(next) == 0:
						send_energy = 0
					else:
						if len(next) == 1:
							if device == self.particle_swarm[workflow, i, next[0]]:
								send_energy = 0
							else:
								if self.env.device_type[self.particle_swarm[workflow, i, next[0]]] == 'CloudServer':
									send_energy = outputSize / WAN * self.env.end_device['sendPower']
								else:
									send_energy = outputSize / LAN * self.env.end_device['sendPower']
						else:
							max_send_time = []
							for y in next:
								next_inputSize = self.env.workflow_task_inputSize[workflow, y]
								if device == self.particle_swarm[workflow, i, y]:
									max_send_time.append(0)
								else:
									if self.env.device_type[self.particle_swarm[workflow, i, y]] == 'CloudServer':
										max_send_time.append(next_inputSize / WAN)
									else:
										max_send_time.append(next_inputSize / LAN)
							send_energy = max(max_send_time) * self.env.end_device['sendPower']

					# 计算总能耗
					device_energy[i, j] += proc_energy + recv_energy + send_energy

		return device_energy

	def get_total_time(self, workflow: int):
		'''
		计算总时间
		:param workflow:
		:return:
		'''
		time = self.get_time(workflow)
		total_time = np.zeros(self.particle_num)
		# 找到最后一个任务
		end = -1
		for i in range(self.env.task_num):
			if self.env.task_next[i] == []:
				end = i
				break
		if end == -1:
			print('error')
			exit(-1)

		# 开始计算总时间
		for i in range(self.particle_num):
			_end = end  # 每个粒子在算的时候都要找到最后一个任务
			while True:
				total_time[i] += time[i, _end]
				# 判断是否到第一个结点
				if self.env.task_prior[_end] == []:
					break
				# 开始递归：寻找前驱结点中最晚完成的
				max = -1
				for j in self.env.task_prior[_end]:
					if max == -1 or time[i, j] > time[i, max]:
						max = j
				_end = max

		return total_time

	def get_total_energy(self, workflow: int):
		'''
		计算终端设备的总能耗
		:param workflow:
		:return:
		'''
		energy = self.get_energy(workflow)
		return np.sum(energy, axis=1)  # 把所有任务的能耗直接相加

	def start_pso(self):
		'''
		pso算法开始迭代
		:return:最终的调度方案、适应度值、时间以及能耗
		'''
		for i in range(self.env.workflow_num):
			print('第%d个工作流' % (i + 1))
			for j in range(self.iter_num):
				print('开始第%d次迭代' % (j + 1))
				self.update_best(i)
				self.update_global_best(i)
				self.update_velocity(i)
				self.update_position(i)

		# 开始返回调度结果
		result = np.full_like(self.env.workflow_task_workload, -1, np.int)  # 调度方案
		fitness = np.full(self.env.workflow_num, -1, np.float)  # 适应度值
		time = np.zeros_like(self.env.workflow_task_workload)  # 时间
		energy = np.zeros_like(self.env.workflow_task_workload)  # 能耗
		for i in range(self.env.workflow_num):
			best = self.global_best_particle[i]
			fitness[i] = self.fitness_value[i, best]
			time[i] = self.get_time(i)[best]
			energy[i] = self.get_energy(i)[best]
			for j in range(self.env.task_num):
				result[i, j] = self.best_particle[i, best, j]

		return result, fitness, time, energy
