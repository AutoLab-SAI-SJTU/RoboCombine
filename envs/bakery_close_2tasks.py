from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class bakery_close_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        basket_pose = rand_pose(xlim=[0.15, 0.2], ylim=[-0.2, -0.15], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
        self.breadbasket_id = np.random.choice([0, 1, 2, 3, 4])
        self.breadbasket = create_actor(scene=self, pose=basket_pose, modelname='076_breadbasket', convex=True, model_id=self.breadbasket_id)
        breadbasket_pose = self.breadbasket.get_pose()
        for _ in range(50):
            bread_pose = rand_pose(xlim=[0.2, 0.28], ylim=[0.0, 0.1], qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=True, rotate_lim=[0, np.pi / 4, 0])
            if (bread_pose.p[0] - breadbasket_pose.p[0]) ** 2 + (bread_pose.p[1] - breadbasket_pose.p[1]) ** 2 > 0.04:
                break
        self.bread_id = np.random.choice([0, 1, 3, 5, 6])
        self.bread = create_actor(scene=self, pose=bread_pose, modelname='075_bread', convex=True, model_id=self.bread_id)
        self.breadbasket.set_mass(0.3)
        self.bread.set_mass(0.05)
        self.add_prohibit_area(self.bread, padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)
        self.switch_model_id = np.random.randint(0, 8)
        self.switch = rand_create_sapien_urdf_obj(scene=self, modelname='056_switch', modelid=self.switch_model_id, xlim=[-0.25, -0.05], ylim=[0.0, 0.1], zlim=[0.81, 0.84], rotate_rand=True, rotate_lim=[0, 0, np.pi / 4], qpos=[0.704141, 0, 0, 0.71006], fix_root_link=True)
        self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2])

    def play_once(self):
        print('[Subtask 1] Place bread in basket')
        arm_tag_bread = ArmTag('right')
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag_bread, pre_grasp_dis=0.07))
        self.move(self.move_by_displacement(arm_tag=arm_tag_bread, z=0.1, move_axis='arm'))
        breadbasket_fp = self.breadbasket.get_functional_point(0)
        if breadbasket_fp is None:
            breadbasket_fp = self.breadbasket.get_pose()
        self.move(self.place_actor(self.bread, arm_tag=arm_tag_bread, target_pose=breadbasket_fp, constrain='free', pre_dis=0.12))
        self.move(self.move_by_displacement(arm_tag=arm_tag_bread, z=0.15, move_axis='arm'))
        print('[Subtask 2] Turn switch')
        arm_tag_switch = ArmTag('left')
        self.move(self.close_gripper(arm_tag=arm_tag_switch, pos=0))
        self.move(self.grasp_actor(self.switch, arm_tag=arm_tag_switch, pre_grasp_dis=0.04))
        self.move(self.open_gripper(arm_tag=arm_tag_switch))
        self.move(self.move_by_displacement(arm_tag_switch, z=0.08, move_axis='arm'))
        self.info['info'] = {'{A}': f'076_breadbasket/base{self.breadbasket_id}', '{B}': f'075_bread/base{self.bread_id}', '{a}': str(arm_tag_bread), '{C}': f'056_switch/base{self.switch_model_id}', '{b}': str(arm_tag_switch)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            breadbasket_pose = self.breadbasket.get_pose().p
            bread_pose = self.bread.get_pose().p
            eps1 = 0.05
            xy_close = np.all(abs(bread_pose[:2] - breadbasket_pose[:2]) < np.array([eps1, eps1]))
            z_lifted = bread_pose[2] > 0.73 + self.table_z_bias
            if xy_close and z_lifted:
                print('  Subtask 1 completed: Bread placed in basket')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            limit = self.switch.get_qlimits()[0]
            if self.switch.get_qpos()[0] >= limit[1] - 0.05:
                print('  Subtask 2 completed: Switch turned')
                self.subtask_completed[1] = True
        left_gripper_val = self.robot.left_gripper_val
        right_gripper_val = self.robot.right_gripper_val
        gripper_check_mode = getattr(self, 'gripper_check_mode', 'full')
        if gripper_check_mode == 'none':
            left_open = True
            right_open = True
            gripper_open = True
        elif gripper_check_mode == 'relaxed':
            left_open = left_gripper_val > 0.6
            right_open = right_gripper_val > 0.05
            gripper_open = left_open and right_open
        else:
            left_open = self.robot.is_left_gripper_open()
            right_open = self.robot.is_right_gripper_open()
            gripper_open = left_open and right_open
        current_step = self.take_action_cnt
        success = all(self.subtask_completed) and gripper_open
        if not hasattr(self, '_last_logged_step'):
            self._last_logged_step = -1
        if current_step != self._last_logged_step:
            self._last_logged_step = current_step
            mode_indicator = {'full': '', 'relaxed': '[R]', 'none': '[N]'}[gripper_check_mode]
            print(f'[Step {current_step:4d}]{mode_indicator} S1:{(chr(10003) if self.subtask_completed[0] else chr(10007))} S2:{(chr(10003) if self.subtask_completed[1] else chr(10007))} L:{left_gripper_val:.3f}({(chr(10003) if left_open else chr(10007))}) R:{right_gripper_val:.3f}({(chr(10003) if right_open else chr(10007))}) -> ', end='')
            if success:
                print('SUCCESS!')
            else:
                reasons = []
                if not self.subtask_completed[0]:
                    reasons.append('S1')
                if not self.subtask_completed[1]:
                    reasons.append('S2')
                if all(self.subtask_completed) and gripper_check_mode != 'none':
                    if not left_open:
                        reasons.append(f"L<{(0.6 if gripper_check_mode == 'relaxed' else 0.8)}")
                    if not right_open:
                        reasons.append(f"R<{(0.05 if gripper_check_mode == 'relaxed' else 0.8)}")
                print(f"FAIL: {','.join(reasons)}")
        return success
