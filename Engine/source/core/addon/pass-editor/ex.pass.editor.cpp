#include "ex.pass.editor.hpp"
#include <se.editor.hpp>

InspectorPass::InspectorPass() { init("./shaders/editor/geometry-viewer-rt.slang",
	{{"SRENDERER_SCENE_TLAS_COUNT", "1"}}); }
	
auto InspectorPass::reflect(rdg::PassReflection& reflector) noexcept -> rdg::PassReflection {
	reflector.add_output("Color").is_texture()
		.with_format(rhi::TextureFormat::RGBA32_FLOAT)
		.consume_as_storage_binding_in_compute();
	reflector.add_output("ChosePoint").is_buffer()
		.with_size(16 * sizeof(float))
		.with_usages(rhi::BufferUsageEnum::STORAGE)
		.with_memory_properties(
			rhi::MemoryPropertyEnum::HOST_VISIBLE_BIT |
			rhi::MemoryPropertyEnum::HOST_COHERENT_BIT);
	return reflector;
}

auto InspectorPass::execute(
	rdg::RenderContext* rdrCtx,
	rdg::RenderData const& rdrDat
) noexcept -> void {
	auto color = rdrDat.get_texture("Color");
	auto buffer = rdrDat.get_buffer("ChosePoint");
	if (buffer->m_buffer->m_mappedData == nullptr) {
		buffer->m_buffer->map_async(0).wait();
	}
	m_interaction = (Interaction*)buffer->m_buffer->get_mapped_range();

	auto scene = rdrDat.get_scene();
	update_binding_scene(rdrCtx, scene);
	update_bindings(rdrCtx, {
		{ "se_scene_tlas", scene->gpu_scene()->binding_resource_tlas() },
		{ "rw_output", rhi::BindingResource(color->get_uav(0, 0, 1)) },
		{ "rw_chose_point", buffer->get_binding_resource() }
		});

	pConst.randomSeed = Random::uniform_int(0, 100000);
	ivec2 mouse_pos = Singleton<editor::EditorContext>::instance()->m_inspector.m_mouseOffset;
	pConst.mousePixelX = mouse_pos;

	auto encoder = begin_pass(rdrCtx);
	encoder->push_constants(&pConst, rhi::ShaderStageEnum::COMPUTE, 0, sizeof(PushConstant));
	encoder->dispatch_workgroups(
		int((color->width() + 31) / 32),
		int((color->height() + 3) / 4), 1);
	encoder->end();

	// update
	const ImGuiIO& IO = ImGui::GetIO();
	pConst.mouseState = 0;
	auto& inspector_data = Singleton<editor::EditorContext>::instance()->m_inspector;
	if (inspector_data.m_hovered && inspector_data.m_focused) {
		if (IO.MouseDown[ImGuiMouseButton_Left]) {
			pConst.mouseState = 1;
		}
		if (IO.MouseReleased[ImGuiMouseButton_Left]) {
			pConst.mouseState = 2;
			m_helper.releaseCountDown = 3;
		}
	}

	if (m_helper.releaseCountDown >= 0) {
		m_helper.releaseCountDown--;
		if (m_helper.releaseCountDown == 0) {
			Singleton<editor::EditorContext>::instance()->m_sceneDisplayed
				->get()->open_node_with_geometry_index(m_interaction->geometryID);
			pConst.hightlightGeometry = m_interaction->geometryID;
		}
	}
}

auto InspectorPass::render_ui() noexcept -> void {
	if (ImGui::BeginTable("DisplayTable", 2, ImGuiTableFlags_None)) {
		// Column setup MUST be here, before any rows
			// First column (labels) - fixed width
		ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
		// Second column (widgets) - stretch to remaining width
		ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

		helper::DrawLabeledControl("Display Mode", [&]() {
			static std::vector<std::string> display_items = {
				"Path Tracing",
				"Albedo",
				"Geometry Normal",
				"Shading Normal",
			};
		ImGui::PushItemWidth(-FLT_MIN);
		if (ImGui::BeginCombo("##Display", display_items[pConst.displayID].c_str())) {
			for (int i = 0; i < display_items.size(); ++i) {
				bool isSelected = (i == pConst.displayID);
				if (ImGui::Selectable(display_items[i].c_str(), isSelected)) {
					pConst.displayID = i;
				}
			}
			ImGui::EndCombo();
		}

		ImGui::PopItemWidth();
		return false;
		});
		
		ImGui::EndTable();
	}

	ImGui::SeparatorText("Hit point");
	if (ImGui::BeginTable("HitpointTable", 2, ImGuiTableFlags_None)) {
		// Column setup MUST be here, before any rows
			// First column (labels) - fixed width
		ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
		// Second column (widgets) - stretch to remaining width
		ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

		helper::DrawLabeledControl("Primitive", [&]() {
			static std::vector<std::string> display_items = {
				"None",
				"Triangle Mesh",
				"Sphere",
				"Rectangle",
				"Box",
			};
			ImGui::PushItemWidth(-FLT_MIN);
			ImGui::Text(display_items[m_interaction->primitiveType + 1].c_str());
			ImGui::PopItemWidth();
			return false;
		});
		helper::DrawLabeledControl("GeometryID", [&]() {
			ImGui::PushItemWidth(-FLT_MIN);
			ImGui::Text("%u", m_interaction->geometryID);
			ImGui::PopItemWidth();
			return false;
		});
		ImGui::EndTable();
	}
}

auto SecondaryInspectorPass::reflect(rdg::PassReflection& reflector) noexcept -> rdg::PassReflection {
	reflector.add_output("Color").is_texture()
		.with_format(rhi::TextureFormat::RGBA32_FLOAT)
		.consume_as_storage_binding_in_compute();
	reflector.add_input("ChosePoint").is_buffer()
		.with_usages(rhi::BufferUsageEnum::STORAGE);
	return reflector;
}

auto SecondaryInspectorPass::execute(
	rdg::RenderContext* rdrCtx,
	rdg::RenderData const& rdrDat
) noexcept -> void {
	auto color = rdrDat.get_texture("Color");
	auto buffer = rdrDat.get_buffer("ChosePoint");

	auto scene = rdrDat.get_scene();
	update_binding_scene(rdrCtx, scene);
	update_bindings(rdrCtx, {
		{ "se_scene_tlas", scene->gpu_scene()->binding_resource_tlas() },
		{ "rw_output", rhi::BindingResource(color->get_uav(0, 0, 1)) },
		{ "rw_chose_point", buffer->get_binding_resource() }
	});

	pConst.randomSeed = Random::uniform_int(0, 100000);

	auto encoder = begin_pass(rdrCtx);
	//encoder->push_constants(&pConst, rhi::ShaderStageEnum::COMPUTE, 0, sizeof(PushConstant));
	encoder->dispatch_workgroups(
		int((color->width() + 31) / 32),
		int((color->height() + 3) / 4), 1);
	encoder->end();
}

auto SecondaryInspectorPass::render_ui() noexcept -> void {
	if (ImGui::BeginTable("DisplayTable", 2, ImGuiTableFlags_None)) {
		// Column setup MUST be here, before any rows
			// First column (labels) - fixed width
		ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 100);
		// Second column (widgets) - stretch to remaining width
		ImGui::TableSetupColumn("Widget", ImGuiTableColumnFlags_WidthStretch);

		helper::DrawLabeledControl("Display Mode", [&]() {
			static std::vector<std::string> display_items = {
				"Path Tracing",
				"Albedo",
				"Geometry Normal",
				"Shading Normal",
			};
		ImGui::PushItemWidth(-FLT_MIN);
		if (ImGui::BeginCombo("##Display", display_items[pConst.displayID].c_str())) {
			for (int i = 0; i < display_items.size(); ++i) {
				bool isSelected = (i == pConst.displayID);
				if (ImGui::Selectable(display_items[i].c_str(), isSelected)) {
					pConst.displayID = i;
				}
			}
			ImGui::EndCombo();
		}

		ImGui::PopItemWidth();
		return false;
			});

		ImGui::EndTable();
	}
}
