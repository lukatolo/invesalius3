import os

import vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPointPicker,
    vtkPolyDataMapper,
    vtkProperty,
    vtkPropPicker,
    vtkRenderer,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals

import invesalius.constants as const
import invesalius.data.polydata_utils as pu
import invesalius.data.coregistration as dcr
import invesalius.data.vtk_utils as vtku
from invesalius import inv_paths
import invesalius.session as ses
from invesalius.pubsub import pub as Publisher

from scipy.spatial import distance


class TargetCoilVisualizer:
    """
    A class for visualizing a single coil during target mode. This is a minimal version of CoilVisualizer
    This only exists during navigation
    """

    # Color for highlighting a marker.
    HIGHLIGHT_COLOR = vtk.vtkNamedColors().GetColor3d("Red")

    # Color for the marker for target when the coil at the target.
    COIL_AT_TARGET_COLOR = vtk.vtkNamedColors().GetColor3d("Green")

    def __init__(
        self,
        parent,
        coil,
        interactor,
        renderer,
        target_guide_renderer,
        actor_factory,
        vector_field_visualizer,
    ):
        self.parent = parent  # TargetCoilPanel class that uses this visualizer
        self.interactor = interactor

        self.renderer = renderer
        self.target_guide_renderer = target_guide_renderer

        # The actor factory is used to create actors for the coil and coil center.
        self.actor_factory = actor_factory

        # The vector field visualizer is used to show a vector field relative to the coil.
        self.vector_field_visualizer = vector_field_visualizer

        self.coil = coil  # This is a dict containing coil name, path, fiducials, etc.
        self.AddCoil(coil)

        # The assembly for showing the vector field relative to the coil in the volume viewer.
        self.vector_field_assembly = self.vector_field_visualizer.CreateVectorFieldAssembly()

        # Add the vector field assembly to the renderer, but make it invisible until the coil is shown.
        self.renderer.AddActor(self.vector_field_assembly)
        self.vector_field_assembly.SetVisibility(0)

        self.aim_actor = None
        self.target_coord = None
        self.coil_at_target = False

        session = ses.Session()
        self.distance_threshold = session.GetConfig(
            "distance_threshold", const.DEFAULT_DISTANCE_THRESHOLD
        )
        self.angle_threshold = session.GetConfig("angle_threshold", const.DEFAULT_ANGLE_THRESHOLD)

        # self.LoadConfig() #LUKATODO: this is low priority and can be implemented after everything is done

        ### copied from viewer_volume.py:EnableTargetMode ###
        # Set viewports to separate the target guide from the volume.
        self.renderer.SetViewport(0, 0, 0.75, 1)
        self.target_guide_renderer.SetViewport(0.75, 0, 1, 1)

        self.distance_text = self.CreateDistanceText()
        self.renderer.AddActor(self.distance_text.actor)
        self.renderer.ResetCamera()

        self.CreateTargetGuide()

        self.target_guide_renderer.ResetCamera()
        self.target_guide_renderer.GetActiveCamera().Zoom(2)
        self.target_guide_renderer.InteractiveOff()
        self.ShowTargetGuide(
            show=False
        )  # Hide the guide by default, hide it only after ResetCamera() is called!
        self.interactor.Render()
        ######################################################################

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.UpdateCoilPoses, "Update coil poses")
        Publisher.subscribe(self.ShowCoil, "Show coil in viewer volume")

    def SetTarget(self, target_coord, m_target):
        self.target_coord = target_coord

        # This function doubles as UnsetTarget (when target_coord is None)
        target_set = target_coord is not None
        self.ShowTargetGuide(show=target_set)

        # Remove old aim_actor
        if self.aim_actor is not None:
            self.renderer.RemoveActor(self.aim_actor)

        if target_set:
            self.AddTargetCoil(m_target)
            self.aim_actor = self.actor_factory.CreateAim(
                target_coord[:3], target_coord[3:], self.HIGHLIGHT_COLOR, scale=1.0
            )
            self.renderer.AddActor(self.aim_actor)
        else:  # unset target
            self.RemoveTargetCoil()
            self.aim_actor = None

    def CreateDistanceText(self):
        distance_text = vtku.Text()

        distance_text.SetSize(const.TEXT_SIZE_DISTANCE_DURING_NAVIGATION)
        distance_text.SetPosition((const.X, 1.0 - const.Y))
        distance_text.SetVerticalJustificationToBottom()
        distance_text.BoldOn()

        return distance_text

    def CreateTargetGuide(self):
        # Using default coil for target guide model as using self.coil_path can cause custom models to overlap.
        coil_path = os.path.join(inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")
        obj_polydata = vtku.CreateObjectPolyData(coil_path)

        normals = vtkPolyDataNormals()
        normals.SetInputData(obj_polydata)
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        mapper.ScalarVisibilityOff()
        # mapper.ImmediateModeRenderingOn()  # improve performance

        obj_roll = vtkActor()
        obj_roll.SetMapper(mapper)
        obj_roll.GetProperty().SetColor(1, 1, 1)
        # obj_roll.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d('GhostWhite'))
        # obj_roll.GetProperty().SetSpecular(30)
        # obj_roll.GetProperty().SetSpecularPower(80)
        obj_roll.SetPosition(0, 25, -30)
        obj_roll.RotateX(-60)
        obj_roll.RotateZ(180)

        obj_yaw = vtkActor()
        obj_yaw.SetMapper(mapper)
        obj_yaw.GetProperty().SetColor(1, 1, 1)
        # obj_yaw.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d('GhostWhite'))
        # obj_yaw.GetProperty().SetSpecular(30)
        # obj_yaw.GetProperty().SetSpecularPower(80)
        obj_yaw.SetPosition(0, -115, 5)
        obj_yaw.RotateZ(180)

        obj_pitch = vtkActor()
        obj_pitch.SetMapper(mapper)
        obj_pitch.GetProperty().SetColor(1, 1, 1)
        # obj_pitch.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d('GhostWhite'))
        # obj_pitch.GetProperty().SetSpecular(30)
        # obj_pitch.GetProperty().SetSpecularPower(80)
        obj_pitch.SetPosition(5, -265, 5)
        obj_pitch.RotateY(90)
        obj_pitch.RotateZ(180)

        arrow_roll_z1 = self.actor_factory.CreateArrow([-50, -35, 12], [-50, -35, 50])
        arrow_roll_z1.GetProperty().SetColor(1, 1, 0)
        arrow_roll_z1.RotateX(-60)
        arrow_roll_z1.RotateZ(180)
        arrow_roll_z2 = self.actor_factory.CreateArrow([50, -35, 0], [50, -35, -50])
        arrow_roll_z2.GetProperty().SetColor(1, 1, 0)
        arrow_roll_z2.RotateX(-60)
        arrow_roll_z2.RotateZ(180)

        arrow_yaw_y1 = self.actor_factory.CreateArrow([-50, -35, 0], [-50, 5, 0])
        arrow_yaw_y1.GetProperty().SetColor(0, 1, 0)
        arrow_yaw_y1.SetPosition(0, -150, 0)
        arrow_yaw_y1.RotateZ(180)
        arrow_yaw_y2 = self.actor_factory.CreateArrow([50, -35, 0], [50, -75, 0])
        arrow_yaw_y2.GetProperty().SetColor(0, 1, 0)
        arrow_yaw_y2.SetPosition(0, -150, 0)
        arrow_yaw_y2.RotateZ(180)

        arrow_pitch_x1 = self.actor_factory.CreateArrow([0, 65, 38], [0, 65, 68])
        arrow_pitch_x1.GetProperty().SetColor(1, 0, 0)
        arrow_pitch_x1.SetPosition(0, -300, 0)
        arrow_pitch_x1.RotateY(90)
        arrow_pitch_x1.RotateZ(180)
        arrow_pitch_x2 = self.actor_factory.CreateArrow([0, -55, 5], [0, -55, -30])
        arrow_pitch_x2.GetProperty().SetColor(1, 0, 0)
        arrow_pitch_x2.SetPosition(0, -300, 0)
        arrow_pitch_x2.RotateY(90)
        arrow_pitch_x2.RotateZ(180)

        self.guide_coil_actors = obj_roll, obj_yaw, obj_pitch
        self.guide_arrow_actors = (
            arrow_roll_z1,
            arrow_roll_z2,
            arrow_yaw_y1,
            arrow_yaw_y2,
            arrow_pitch_x1,
            arrow_pitch_x2,
        )
        for ind in self.guide_coil_actors:
            self.target_guide_renderer.AddActor(ind)

        for ind in self.guide_arrow_actors:
            self.target_guide_renderer.AddActor(ind)

    def ShowTargetGuide(self, show=True):
        self.distance_text.Show(show)
        for actor in self.target_guide_renderer.GetActors():
            actor.SetVisibility(show)

    def UpdateVectorField(self):
        """
        Update the vector field assembly to reflect the current vector field.
        """
        # Create a new vector field assembly.
        new_vector_field_assembly = self.vector_field_visualizer.CreateVectorFieldAssembly()

        # Replace the old vector field assembly with the new one.
        self.actor_factory.ReplaceActor(
            self.renderer, self.vector_field_assembly, new_vector_field_assembly
        )

        # Store the new vector field assembly.
        self.vector_field_assembly = new_vector_field_assembly

    def SetCoilAtTarget(self, state):
        self.coil_at_target = state

        vtk_colors = vtk.vtkNamedColors()

        # Set the color of the target coil based on whether the coil is at the target or not.
        target_coil_color = (
            vtk_colors.GetColor3d("Green") if state else vtk_colors.GetColor3d("DarkOrange")
        )

        # Set the color of both target coil (representing the target) and the coil center (representing the actual coil).
        self.coil["target_actor"].GetProperty().SetDiffuseColor(target_coil_color)
        self.coil["center_actor"].GetProperty().SetDiffuseColor(target_coil_color)

        if (
            state and self.aim_actor is not None
        ):  # Only make aim_actor is green if state else red (color when initialized)
            self.aim_actor.GetProperty().SetDiffuseColor(target_coil_color)

    def ShowCoil(self, state, coil_name=None):
        coil = self.coil
        coil["actor"].SetVisibility(state)
        # Always show the center donut actor and target
        # coil["center_actor"].SetVisibility(True)
        self.vector_field_assembly.SetVisibility(state)

    def AddTargetCoil(self, m_target):
        self.RemoveTargetCoil()

        vtk_colors = vtk.vtkNamedColors()

        # LUKATODO: this is an arbitrary coil... but works for single coil mode
        decoded_path = self.coil["path"]

        coil_filename = os.path.basename(decoded_path)
        coil_dir = os.path.dirname(decoded_path)

        # A hack to load the coil without the handle for the Magstim figure-8 coil.
        coil_path = (
            os.path.join(coil_dir, coil_filename)
            if coil_filename != "magstim_fig8_coil.stl"
            else os.path.join(coil_dir, "magstim_fig8_coil_no_handle.stl")
        )

        obj_polydata = vtku.CreateObjectPolyData(coil_path)

        transform = vtk.vtkTransform()
        transform.RotateZ(90)

        transform_filt = vtk.vtkTransformPolyDataFilter()
        transform_filt.SetTransform(transform)
        transform_filt.SetInputData(obj_polydata)
        transform_filt.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(transform_filt.GetOutput())
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.Update()

        obj_mapper = vtk.vtkPolyDataMapper()
        obj_mapper.SetInputData(normals.GetOutput())
        obj_mapper.ScalarVisibilityOff()
        # obj_mapper.ImmediateModeRenderingOn()  # improve performance

        target_actor = vtk.vtkActor()
        target_actor.SetMapper(obj_mapper)
        target_actor.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d("DarkOrange"))
        target_actor.GetProperty().SetSpecular(0.5)
        target_actor.GetProperty().SetSpecularPower(10)
        target_actor.GetProperty().SetOpacity(0.3)
        target_actor.SetVisibility(True)
        target_actor.SetUserMatrix(m_target)

        self.renderer.AddActor(target_actor)

        self.coil["target_actor"] = target_actor

    def RemoveTargetCoil(self):
        if self.coil.get("target_actor") is None:
            return

        self.renderer.RemoveActor(self.coil["target_actor"])
        self.coil["target_actor"] = None

    def AddCoil(self, coil):
        """
        Add actors for actual coil, coil center, and x, y, and z-axes to the renderer.
        """
        coil_path = coil["path"]

        vtk_colors = vtk.vtkNamedColors()
        obj_polydata = vtku.CreateObjectPolyData(coil_path)

        transform = vtk.vtkTransform()
        transform.RotateZ(90)

        transform_filt = vtk.vtkTransformPolyDataFilter()
        transform_filt.SetTransform(transform)
        transform_filt.SetInputData(obj_polydata)
        transform_filt.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(transform_filt.GetOutput())
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.Update()

        obj_mapper = vtk.vtkPolyDataMapper()
        obj_mapper.SetInputData(normals.GetOutput())
        obj_mapper.ScalarVisibilityOff()
        # obj_mapper.ImmediateModeRenderingOn()  # improve performance?

        coil_actor = vtk.vtkActor()
        coil_actor.SetMapper(obj_mapper)
        coil_actor.GetProperty().SetAmbientColor(vtk_colors.GetColor3d("GhostWhite"))
        coil_actor.GetProperty().SetSpecular(30)
        coil_actor.GetProperty().SetSpecularPower(80)
        coil_actor.GetProperty().SetOpacity(0.4)
        coil_actor.SetVisibility(1)

        # Create an actor for the coil center.
        coil_center_actor = self.actor_factory.CreateTorus(
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0],
            colour=vtk_colors.GetColor3d("Red"),
            scale=0.5,
        )

        self.renderer.AddActor(coil_actor)
        self.renderer.AddActor(coil_center_actor)

        self.coil["actor"] = coil_actor
        self.coil["center_actor"] = coil_center_actor

        # LUKATODO: Vector field assembly follows a different pattern for addition, should unify.
        # self.vector_field_assembly.SetVisibility(1)

    def RemoveCoil(self):
        coil = self.coil
        self.renderer.RemoveActor(coil["actor"])
        self.renderer.RemoveActor(coil["center_actor"])
        coil["actor"] = None
        coil["center_actor"] = None
        # self.vector_field_assembly.SetVisibility(0)

    def UpdateCoilPoses(self, m_imgs, coords):
        """
        During navigation, use updated coil pose to perform the following tasks:

        - Update actor positions for coil, coil center, and coil orientation axes.
        """

        # Use the m_img relevant to this coil
        m_img_flip = m_imgs[self.coil["name"]].copy()
        m_img_flip[1, -1] = -m_img_flip[1, -1]
        m_img_vtk = vtku.numpy_to_vtkMatrix4x4(m_img_flip)

        # Update actor positions for coil, coil center, and coil orientation axes.
        self.coil["actor"].SetUserMatrix(m_img_vtk)
        self.coil["center_actor"].SetUserMatrix(m_img_vtk)

        # Update target_guide
        self.UpdateTargetGuide(m_imgs[self.coil["name"]], coords[self.coil["name"]])

        self.interactor.Render()
        # LUKATODO
        # self.vector_field_assembly.SetUserMatrix(m_img_vtk)

    # Copied from viewer_volume.py:OnUpdateCoilPose
    def UpdateTargetGuide(self, m_img, coord):
        if self.target_coord is not None:
            distance_to_target = distance.euclidean(
                coord[0:3], (self.target_coord[0], -self.target_coord[1], self.target_coord[2])
            )

            formatted_distance = f"Distance: {distance_to_target: >5.1f} mm"

            self.distance_text.SetValue(formatted_distance)

            self.renderer.ResetCamera()
            self.parent.SetCameraTarget()
            if distance_to_target > 100:
                distance_to_target = 100
            # ((-0.0404*dst) + 5.0404) is the linear equation to normalize the zoom between 1 and 5 times with
            # the distance between 1 and 100 mm
            self.renderer.GetActiveCamera().Zoom((-0.0404 * distance_to_target) + 5.0404)

            is_under_distance_threshold = distance_to_target <= self.distance_threshold

            m_img_flip = m_img.copy()
            m_img_flip[1, -1] = -m_img_flip[1, -1]

            # Send displacement to the robot. # LUKATODO: do this elsewhere?
            displacement_to_target_robot = dcr.ComputeRelativeDistanceToTarget(
                target_coord=self.target_coord, m_img=m_img_flip
            )
            # wx.CallAfter(
            #     Publisher.sendMessage,
            #     "Neuronavigation to Robot: Update displacement to target",
            #     displacement=displacement_to_target_robot,
            # )

            distance_to_target = displacement_to_target_robot.copy()
            if distance_to_target[3] > const.ARROW_UPPER_LIMIT:
                distance_to_target[3] = const.ARROW_UPPER_LIMIT
            elif distance_to_target[3] < -const.ARROW_UPPER_LIMIT:
                distance_to_target[3] = -const.ARROW_UPPER_LIMIT
            coordrx_arrow = const.ARROW_SCALE * distance_to_target[3]

            if distance_to_target[4] > const.ARROW_UPPER_LIMIT:
                distance_to_target[4] = const.ARROW_UPPER_LIMIT
            elif distance_to_target[4] < -const.ARROW_UPPER_LIMIT:
                distance_to_target[4] = -const.ARROW_UPPER_LIMIT
            coordry_arrow = const.ARROW_SCALE * distance_to_target[4]

            if distance_to_target[5] > const.ARROW_UPPER_LIMIT:
                distance_to_target[5] = const.ARROW_UPPER_LIMIT
            elif distance_to_target[5] < -const.ARROW_UPPER_LIMIT:
                distance_to_target[5] = -const.ARROW_UPPER_LIMIT
            coordrz_arrow = const.ARROW_SCALE * distance_to_target[5]

            if self.guide_arrow_actors is not None:
                for actor in self.guide_arrow_actors:
                    self.target_guide_renderer.RemoveActor(actor)

            if (
                self.angle_threshold * const.ARROW_SCALE
                > coordrx_arrow
                > -self.angle_threshold * const.ARROW_SCALE
            ):
                is_under_x_angle_threshold = True
                self.guide_coil_actors[0].GetProperty().SetColor(0, 1, 0)
            else:
                is_under_x_angle_threshold = False
                self.guide_coil_actors[0].GetProperty().SetColor(1, 1, 1)

            offset = 5

            arrow_roll_x1 = self.actor_factory.CreateArrow(
                [-55, -35, offset], [-55, -35, offset - coordrx_arrow]
            )
            arrow_roll_x1.RotateX(-60)
            arrow_roll_x1.RotateZ(180)
            arrow_roll_x1.GetProperty().SetColor(1, 1, 0)

            arrow_roll_x2 = self.actor_factory.CreateArrow(
                [55, -35, offset], [55, -35, offset + coordrx_arrow]
            )
            arrow_roll_x2.RotateX(-60)
            arrow_roll_x2.RotateZ(180)
            arrow_roll_x2.GetProperty().SetColor(1, 1, 0)

            if (
                self.angle_threshold * const.ARROW_SCALE
                > coordrz_arrow
                > -self.angle_threshold * const.ARROW_SCALE
            ):
                is_under_z_angle_threshold = True
                self.guide_coil_actors[1].GetProperty().SetColor(0, 1, 0)
            else:
                is_under_z_angle_threshold = False
                self.guide_coil_actors[1].GetProperty().SetColor(1, 1, 1)

            offset = -35

            arrow_yaw_z1 = self.actor_factory.CreateArrow(
                [-55, offset, 0], [-55, offset - coordrz_arrow, 0]
            )
            arrow_yaw_z1.SetPosition(0, -150, 0)
            arrow_yaw_z1.RotateZ(180)
            arrow_yaw_z1.GetProperty().SetColor(0, 1, 0)

            arrow_yaw_z2 = self.actor_factory.CreateArrow(
                [55, offset, 0], [55, offset + coordrz_arrow, 0]
            )
            arrow_yaw_z2.SetPosition(0, -150, 0)
            arrow_yaw_z2.RotateZ(180)
            arrow_yaw_z2.GetProperty().SetColor(0, 1, 0)

            if (
                self.angle_threshold * const.ARROW_SCALE
                > coordry_arrow
                > -self.angle_threshold * const.ARROW_SCALE
            ):
                is_under_y_angle_threshold = True
                self.guide_coil_actors[2].GetProperty().SetColor(0, 1, 0)
            else:
                is_under_y_angle_threshold = False
                self.guide_coil_actors[2].GetProperty().SetColor(1, 1, 1)

            offset = 38
            arrow_pitch_y1 = self.actor_factory.CreateArrow(
                [0, 65, offset], [0, 65, offset + coordry_arrow]
            )
            arrow_pitch_y1.SetPosition(0, -300, 0)
            arrow_pitch_y1.RotateY(90)
            arrow_pitch_y1.RotateZ(180)
            arrow_pitch_y1.GetProperty().SetColor(1, 0, 0)

            offset = 5
            arrow_pitch_y2 = self.actor_factory.CreateArrow(
                [0, -55, offset], [0, -55, offset - coordry_arrow]
            )
            arrow_pitch_y2.SetPosition(0, -300, 0)
            arrow_pitch_y2.RotateY(90)
            arrow_pitch_y2.RotateZ(180)
            arrow_pitch_y2.GetProperty().SetColor(1, 0, 0)

            # Combine all the conditions to check if the coil is at the target.
            coil_at_target = (
                is_under_distance_threshold
                and is_under_x_angle_threshold
                and is_under_y_angle_threshold
                and is_under_z_angle_threshold
            )

            self.SetCoilAtTarget(coil_at_target)
            # LUKATODO: robot
            # wx.CallAfter(Publisher.sendMessage, "Coil at target", state=coil_at_target)
            # wx.CallAfter(
            #     Publisher.sendMessage, "From Neuronavigation: Coil at target", state=coil_at_target
            # )

            self.guide_arrow_actors = (
                arrow_roll_x1,
                arrow_roll_x2,
                arrow_yaw_z1,
                arrow_yaw_z2,
                arrow_pitch_y1,
                arrow_pitch_y2,
            )

            for ind in self.guide_arrow_actors:
                self.target_guide_renderer.AddActor(ind)
