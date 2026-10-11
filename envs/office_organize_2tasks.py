from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class office_organize_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        mouse_pos = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi / 4, 0])
        self.mouse_id = np.random.choice([0, 1, 2], 1)[0]
        self.mouse = create_actor(scene=self, pose=mouse_pos, modelname='047_mouse', convex=True, model_id=self.mouse_id)
        self.mouse.set_mass(0.05)
        mousepad_pose = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[1, 0, 0, 0], rotate_rand=False)
        while np.sqrt((mousepad_pose.p[0] - mouse_pos.p[0]) ** 2 + (mousepad_pose.p[1] - mouse_pos.p[1]) ** 2) < 0.1:
            mousepad_pose = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[1, 0, 0, 0], rotate_rand=False)
        mouse_colors = {'Red': (1, 0, 0), 'Green': (0, 1, 0), 'Blue': (0, 0, 1), 'Yellow': (1, 1, 0), 'Cyan': (0, 1, 1), 'Magenta': (1, 0, 1), 'Black': (0, 0, 0), 'Gray': (0.5, 0.5, 0.5)}
        mc_items = list(mouse_colors.items())
        mc_idx = np.random.choice(len(mc_items))
        self.mouse_color_name, mouse_color_val = mc_items[mc_idx]
        self.mousepad = create_box(scene=self, pose=mousepad_pose, half_size=[0.035, 0.065, 0.0005], color=mouse_color_val, name='mousepad_right', is_static=True)
        self.add_prohibit_area(self.mousepad, padding=0.12)
        self.add_prohibit_area(self.mouse, padding=0.03)
        self.mousepad_pose = self.mousepad.get_pose().p.tolist() + [0, 0, 0, 1]
        seal_pos = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.05, 0.05], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False)
        self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]
        self.seal = create_actor(scene=self, pose=seal_pos, modelname='100_seal', convex=True, model_id=self.seal_id)
        self.seal.set_mass(0.05)
        seal_tgt_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.05, 0.05], qpos=[1, 0, 0, 0], rotate_rand=False)
        while np.sqrt((seal_tgt_pose.p[0] - seal_pos.p[0]) ** 2 + (seal_tgt_pose.p[1] - seal_pos.p[1]) ** 2) < 0.1:
            seal_tgt_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.05, 0.1], qpos=[1, 0, 0, 0], rotate_rand=False)
        seal_colors = {'Red': (1, 0, 0), 'Green': (0, 1, 0), 'Blue': (0, 0, 1), 'Yellow': (1, 1, 0), 'Orange': (1, 0.5, 0), 'Purple': (0.5, 0, 0.5)}
        seal_items = list(seal_colors.items())
        seal_cidx = np.random.choice(len(seal_items))
        self.seal_color_name, seal_color_val = seal_items[seal_cidx]
        self.seal_target = create_visual_box(scene=self, pose=seal_tgt_pose, half_size=[0.035, 0.035, 0.0005], color=seal_color_val, name='seal_target_left')
        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.seal_target, padding=0.1)

    def play_once(self):
        print('[Subtask 1] Place mouse on pad')
        arm_tag_mouse = ArmTag('right')
        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag_mouse, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_mouse, z=0.1))
        self.move(self.place_actor(self.mouse, arm_tag=arm_tag_mouse, target_pose=self.mousepad_pose, constrain='align', pre_dis=0.07, dis=0.005))
        print('[Subtask 2] Stamp seal')
        arm_tag_seal = ArmTag('left')
        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag_seal, pre_grasp_dis=0.1, contact_point_id=[4, 5, 6, 7]))
        self.move(self.move_by_displacement(arm_tag=arm_tag_seal, z=0.05))
        self.move(self.place_actor(self.seal, arm_tag=arm_tag_seal, target_pose=self.seal_target.get_pose(), pre_dis=0.1, constrain='auto'))
        self.info['info'] = {'{A}': f'047_mouse/base{self.mouse_id}', '{B}': f'{self.mouse_color_name}', '{a}': str(arm_tag_mouse), '{C}': f'100_seal/base{self.seal_id}', '{D}': f'{self.seal_color_name}', '{b}': str(arm_tag_seal)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            mouse_pose = self.mouse.get_pose().p
            mouse_qpose = np.abs(self.mouse.get_pose().q)
            target_pos = self.mousepad.get_pose().p
            eps1, eps2 = (0.015, 0.012)
            pos_ok = np.all(abs(mouse_pose[:2] - target_pos[:2]) < np.array([eps1, eps2]))
            rot_ok = np.abs(mouse_qpose[2] * mouse_qpose[3] - 0.49) < eps1 or np.abs(mouse_qpose[0] * mouse_qpose[1] - 0.49) < eps1
            if pos_ok and rot_ok:
                print('  Subtask 1 completed: Mouse placed on pad')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            seal_pose = self.seal.get_pose().p
            seal_tgt = self.seal_target.get_pose().p
            eps1 = 0.01
            if np.all(abs(seal_pose[:2] - seal_tgt[:2]) < np.array([eps1, eps1])):
                print('  Subtask 2 completed: Seal stamped on target')
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
