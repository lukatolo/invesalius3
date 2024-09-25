import os

import vtk

import invesalius.constants as const
import invesalius.data.polydata_utils as pu
import invesalius.data.vtk_utils as vtku
import invesalius.session as ses
from invesalius.pubsub import pub as Publisher


class TargetCoilVisualizer:
    """
    A class for visualizing a single coil during target mode. This is a minimal version of CoilVisualizer
    This only exists during navigation
    """

    # Color for highlighting a marker.
    HIGHLIGHT_COLOR = vtk.vtkNamedColors().GetColor3d("Red")

    # Color for the marker for target when the coil at the target.
    COIL_AT_TARGET_COLOR = vtk.vtkNamedColors().GetColor3d("Green")

    def __init__(self, coil, interactor, renderer, actor_factory, vector_field_visualizer):
        self.interactor = interactor
        self.renderer = renderer

        # The actor factory is used to create actors for the coil and coil center.
        self.actor_factory = actor_factory

        # The vector field visualizer is used to show a vector field relative to the coil.
        self.vector_field_visualizer = vector_field_visualizer

        # The full list of keys in the coil dict can be found in the config file
        # But the relevant keys for this class are: [name, path]
        self.coil = coil
        
        # Entries for the following keys [actor, center_actor, target_actor] are added here
        self.AddCoil(coil)
        #self.AddTargetCoil()

        # self.AddTargetCoil() # LUKATODO: get m_target here?

        # The assembly for showing the vector field relative to the coil in the volume viewer.
        self.vector_field_assembly = self.vector_field_visualizer.CreateVectorFieldAssembly()

        # Add the vector field assembly to the renderer, but make it invisible until the coil is shown.
        self.renderer.AddActor(self.vector_field_assembly)
        self.vector_field_assembly.SetVisibility(0)

        self.coil_at_target = False

        # self.LoadConfig() #LUKATODO: this is low priority and can be implemented after everything is done

        self.__bind_events()

    def __bind_events(self):
        #Publisher.subscribe(self.SetCoilAtTarget, "Coil at target")
        Publisher.subscribe(self.UpdateCoilPoses, "Update coil poses")
        Publisher.subscribe(self.UpdateVectorField, "Update vector field") # LUKATODO: this message should carry coil_name with it...

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

    def ShowCoil(self, state):
        coil = self.coil
        coil["actor"].SetVisibility(state)
        coil["center_actor"].SetVisibility(True)  # Always show the center donut actor
        coil["target_actor"].SetVisibility(state)
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
        if (target_actor := self.coil.get("target_actor", None)) is None:
            return

        self.renderer.RemoveActor(target_actor)
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

        self.interactor.Render()
        # LUKATODO
        # self.vector_field_assembly.SetUserMatrix(m_img_vtk)
