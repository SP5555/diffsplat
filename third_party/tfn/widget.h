// ======================================================================== //
// Copyright Qi Wu, since 2019                                              //
// Copyright SCI Institute, University of Utah, 2018                        //
// ======================================================================== //
// Modified by Set Paing, 2026:                                             //
//   - freehand "draw" mode for the alpha curve (paint a dense per-column   //
//     curve directly; Douglas-Peucker simplification back to control       //
//     points when leaving draw mode)                                      //
//   - JSON save/load wired up to real file dialogs, with a raw-curve       //
//     encoding path so a drawn curve round-trips exactly, not just its     //
//     control-point approximation                                         //
//   - color "blueprint" picker: click-to-apply swatches for built-in       //
//     colormaps that only overwrite color control points, leaving the     //
//     alpha curve untouched (replaces the old text dropdown)              //
//   - fixed-height alpha graph so window resizes no longer distort it      //
// ======================================================================== //
#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "core.h"
#include "default.h"

#include <imconfig.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tfn {

class TFN_MODULE_INTERFACE TransferFunctionWidget
{
 private:
  using setter = std::function<void(const list3f &, const list2f &, const vec2f &)>;

 private:
  /* Variables Controlled by Users */
  setter _setter_cb;
  vec2f valueRange; //< the current value range controlled by the user
  vec2f defaultRange; //< the default value range being displayed on the GUI

  /* The 2d palette texture on the GPU for displaying the color map preview in the UI. */
  GLuint tfn_palette;

  // all available transfer functions
  std::vector<std::string> tfns_names;
  std::vector<tfn::TransferFunctionCore> tfns;
  int num_builtin_tfns{0}; // tfns[0, num_builtin_tfns) are the built-in color blueprints
  // Frozen snapshot of each built-in's color stops, captured once at startup.
  // tfns[i] itself gets mutated in place whenever it's the active editing target
  // (current_colorpoints aliases straight into it -- see select_tfn), so the
  // blueprint picker must read from here, never from tfns[i] directly.
  std::vector<std::vector<tfn::TransferFunctionCore::ColorControl>> blueprint_colors;

  using ColorPoint = tfn::TransferFunctionCore::ColorControl;
  using AlphaPoint = tfn::TransferFunctionCore::AlphaControl;
  using GaussianPoint = tfn::TransferFunctionCore::GaussianObject;

  // properties of currently selected transfer function
  int tfn_selection{-1};
  std::vector<ColorPoint>* current_colorpoints{};
  std::vector<AlphaPoint>* current_alphapoints{};
  vec2i current_tfn_editable{1, 1};

  std::vector<AlphaPoint> uneditable_alphapoints;

  // freehand "draw" mode for the alpha curve: paints directly into a dense
  // 256-sample buffer instead of dragging sparse control points.
  bool  alpha_draw_mode{false};
  std::vector<AlphaPoint> draw_alphapoints;      // 256-sample freehand curve buffer
  std::vector<AlphaPoint>* real_alphapoints{};   // always points at the tfn's true control-point vector
  bool  draw_dragging{false};
  float draw_last_col{0.f};
  float draw_last_alpha{0.f};

  // flag indicating transfer function has changed in UI
  bool tfn_changed{true};
  bool tfn_applied{true};

  // scaling factor for generated alphas
  float global_alpha_scale{1.f};

  // domain (value range) of transfer function
  vec2f value_range{-1.f, 1.f};
  vec2f value_range_default{-1.f, 1.f};
  vec2f value_range_percentage{0.f, 100.f};

 public:
  ~TransferFunctionWidget();
  TransferFunctionWidget(const setter &);
  
  /* Setup the default data value range for the transfer function */
  void set_default_value_range(const float &a, const float &b);
  
  /* Draw the transfer function editor widget, returns true if the transfer function changed */
  bool build(bool *p_open = NULL, bool do_not_render_textures = false);
  
  /* Construct the ImGui GUI */
  void build_gui();
  
  /* Render the transfer function to a 1D texture that can be applied to volume data */
  void render(int tfn_w = 256, int tfn_h = 1);

  /* Load the transfer function in the file passed and set it active. Returns false on failure. */
  bool load(const std::string &fileName);

  /* Save the current transfer function out to the file. Returns false on failure. */
  bool save(const std::string &fileName);

  /* Create a new TFN profile */
  // void add_tfn(const tfn::TransferFunctionCore& core, const std::string &name);
  void add_tfn(const list4f &, const list2f &, const std::string &name);
  
 private:
  /* Change selection */
  void select_tfn(int selection);
  /** Load all the pre-defined transfer function maps */
  void set_default_tfns();
  /** Draw the Tfn Editor in a window */
  void draw_tfn_editor(const float margin, const float height);
  tfn::vec4f draw_tfn_editor__preview_texture(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &);
  tfn::vec4f draw_tfn_editor__color_control_points(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &);
  tfn::vec4f draw_tfn_editor__alpha_control_points(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &);
  tfn::vec4f draw_tfn_editor__alpha_freehand(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &);
  tfn::vec4f draw_tfn_editor__interaction_blocks(void *_draw_list, const tfn::vec3f &, const tfn::vec2f &, const tfn::vec4f &, const float &, const float &);

  /* Sample the current control-point curve into the 256-entry draw_alphapoints buffer. */
  void rasterize_alpha_to_draw_curve();
  /* Approximate draw_alphapoints with at most max_points control points (Douglas-Peucker). */
  void simplify_draw_curve_to_controlpoints(int max_points);

  /* Sample a color control-point curve into `samples` evenly-spaced RGB values. */
  std::vector<vec3f> sample_color_gradient(std::vector<ColorPoint> *pts, int samples) const;
  /* Copy blueprint tfns[idx]'s colors onto the currently-edited color control points. Alpha untouched. */
  void apply_color_blueprint(int idx);
  /* Draw the grid of built-in color blueprint swatches. */
  void draw_color_blueprint_picker();
};

inline void TransferFunctionWidget::select_tfn(int selection)
{
  if (tfn_selection != selection) 
  {
    tfn_selection = selection;

    auto& tfn = tfns[tfn_selection];

    current_colorpoints = tfn.colorControlVector();
    current_tfn_editable.x = (tfn.colorControlCount() > 128) ? 0 : 1;

    // in this case we have to use the raw RGBA table
    if (tfn.alphaControlCount() == 0 || tfn.gaussianObjectCount() > 0)
    {
      uneditable_alphapoints.resize(tfn.resolution());
      const auto *table = (vec4f *)tfn.data();
      for (int i = 0; i < uneditable_alphapoints.size(); ++i) {
        uneditable_alphapoints[i] = vec2f((float)i / (uneditable_alphapoints.size() - 1), table[i].w);
      }
      current_alphapoints = &uneditable_alphapoints;
      current_tfn_editable.y = 0;
    }
    else {
      current_alphapoints = tfn.alphaControlVector();
      current_tfn_editable.y = (tfn.alphaControlCount() > 128) ? 0 : 1;
    }

    // switching transfer functions always drops back to control-point mode
    real_alphapoints = current_alphapoints;
    alpha_draw_mode  = false;
    draw_dragging    = false;

    tfn_changed  = true;
  }
}

inline TransferFunctionWidget::~TransferFunctionWidget()
{
  if (tfn_palette) glDeleteTextures(1, &tfn_palette);
}

inline TransferFunctionWidget::TransferFunctionWidget(const setter &fcn)
    : tfn_changed(true), tfn_palette(0), _setter_cb(fcn), valueRange{0.f, 0.f}, defaultRange{0.f, 0.f}
{
  set_default_tfns();
  num_builtin_tfns = (int)tfns.size();
  blueprint_colors.resize(num_builtin_tfns);
  for (int i = 0; i < num_builtin_tfns; ++i)
    blueprint_colors[i] = *tfns[i].colorControlVector();
  select_tfn(0);
}

// inline void TransferFunctionWidget::add_tfn(const tfn::TransferFunctionCore& core, const std::string &name)
// {
//   auto it = std::find(tfns_names.begin(), tfns_names.end(), name);
//   if (it == tfns_names.end()) {
//     tfns.push_back(core);
//     tfns.back().updateColorMap();
//     tfns_names.push_back(name);
//     select_tfn((int)(tfns.size() - 1)); // Remember to update other constructors also
//   } else {
//     select_tfn((int)std::distance(tfns_names.begin(), it));
//   }
// }

inline void TransferFunctionWidget::add_tfn(const list4f &ct, const list2f &ot, const std::string &name)
{
  auto it = std::find(tfns_names.begin(), tfns_names.end(), name);

  if (it == tfns_names.end()) {
    tfns.emplace_back();
    auto& tfn = tfns.back();

    for (size_t i = 0; i < ct.size(); ++i) {
      tfn.addColorControl(ct[i].x, ct[i].y, ct[i].z, ct[i].w);
    }

    for (size_t i = 0; i < ot.size(); ++i) {
      tfn.addAlphaControl(vec2f{ot[i].x, ot[i].y});
    }
    
    tfn.updateColorMap();

    tfns_names.push_back(name);

    select_tfn((int)(tfns.size() - 1)); // Remember to update other constructors also
  } else {
    select_tfn((int)std::distance(tfns_names.begin(), it));
  }
}

inline void TransferFunctionWidget::set_default_value_range(const float &a, const float &b)
{
  if (b >= a) {
    valueRange.x = defaultRange.x = a;
    valueRange.y = defaultRange.y = b;
    tfn_changed = true;
  }
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__preview_texture(void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor)
{
  auto draw_list = (ImDrawList *)_draw_list;
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y));
  ImGui::Image(reinterpret_cast<void *>(tfn_palette), (const ImVec2 &)size);
  ImGui::SetCursorScreenPos((const ImVec2 &)cursor);
  // TODO: more generic way of drawing arbitary splats
  for (int i = 0; i < current_alphapoints->size() - 1; ++i) {
    std::vector<ImVec2> polyline;
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i].pos.x * size.x, cursor.y + size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i].pos.x * size.x, cursor.y + (1.f - (*current_alphapoints)[i].pos.y) * size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i + 1].pos.x * size.x + 1, cursor.y + (1.f - (*current_alphapoints)[i + 1].pos.y) * size.y);
    polyline.emplace_back(cursor.x + margin.x + (*current_alphapoints)[i + 1].pos.x * size.x + 1, cursor.y + size.y);
#ifdef IMGUI_VERSION_NUM
    draw_list->AddConvexPolyFilled(polyline.data(), (int)polyline.size(), 0xFFD8D8D8 /*, true*/);
#else
    draw_list->AddConvexPolyFilled(polyline.data(), (int)polyline.size(), 0xFFD8D8D8, true);
#endif
  }
  tfn::vec4f new_cursor = {
      cursor.x,
      cursor.y + size.y + margin.z,
      cursor.z,
      cursor.w - size.y,
  };
  ImGui::SetCursorScreenPos((const ImVec2 &)new_cursor);
  return new_cursor;
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__color_control_points(void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &color_len)
{
  auto draw_list = (ImDrawList *)_draw_list;
  // draw circle background
  draw_list->AddRectFilled(
      ImVec2(cursor.x + margin.x, cursor.y - margin.z), 
      ImVec2(cursor.x + margin.x + size.x, cursor.y - margin.x + 2.5f * color_len),
      0xFF474646
  );
  // draw circles
  for (int i = (int)current_colorpoints->size() - 1; i >= 0; --i) {
    const ImVec2 pos(cursor.x + size.x * (*current_colorpoints)[i].position + margin.x, cursor.y);
    ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y));
    // white background
    draw_list->AddTriangleFilled(
        ImVec2(pos.x - 0.5f * color_len, pos.y), ImVec2(pos.x + 0.5f * color_len, pos.y), ImVec2(pos.x, pos.y - color_len), 0xFFD8D8D8);
    draw_list->AddCircleFilled(ImVec2(pos.x, pos.y + 0.5f * color_len), color_len, 0xFFD8D8D8);
    // draw picker
    ImVec4 picked_color = ImColor((*current_colorpoints)[i].color.x, (*current_colorpoints)[i].color.y, (*current_colorpoints)[i].color.z, 1.f);
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y + 1.5f * color_len));
    if (ImGui::ColorEdit4(("##ColorPicker" + std::to_string(i)).c_str(),
            (float *)&picked_color,
            ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview
                | ImGuiColorEditFlags_NoOptions | ImGuiColorEditFlags_NoTooltip)) {
      (*current_colorpoints)[i].color.x = picked_color.x;
      (*current_colorpoints)[i].color.y = picked_color.y;
      (*current_colorpoints)[i].color.z = picked_color.z;
      tfn_changed = true;
    }
    if (ImGui::IsItemHovered()) {
      // convert float color to char
      int cr = static_cast<int>(picked_color.x * 255);
      int cg = static_cast<int>(picked_color.y * 255);
      int cb = static_cast<int>(picked_color.z * 255);
      // setup tooltip
      ImGui::BeginTooltip();
      ImVec2 sz(ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2, ImGui::GetFontSize() * 4 + ImGui::GetStyle().FramePadding.y * 2);
      ImGui::ColorButton("##PreviewColor", picked_color, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_AlphaPreview, sz);
      ImGui::SameLine();
      ImGui::Text(
          "Left click to edit\n"
          "HEX: #%02X%02X%02X\n"
          "RGB: [%3d,%3d,%3d]\n(%.2f, %.2f, %.2f)",
          cr,
          cg,
          cb,
          cr,
          cg,
          cb,
          picked_color.x,
          picked_color.y,
          picked_color.z);
      ImGui::EndTooltip();
    }
  }
  for (int i = 0; i < current_colorpoints->size(); ++i) {
    const ImVec2 pos(cursor.x + size.x * (*current_colorpoints)[i].position + margin.x, cursor.y);
    // draw button
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y - 0.5f * color_len));
    ImGui::InvisibleButton(("##ColorControl-" + std::to_string(i)).c_str(), ImVec2(2.f * color_len, 2.f * color_len));
    // dark highlight
    ImGui::SetCursorScreenPos(ImVec2(pos.x - color_len, pos.y));
    draw_list->AddCircleFilled(ImVec2(pos.x, pos.y + 0.5f * color_len), 0.5f * color_len, ImGui::IsItemHovered() ? 0xFF051C33 : 0xFFBCBCBC);
    // delete color point
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
      if (i > 0 && i < current_colorpoints->size() - 1) {
        current_colorpoints->erase(current_colorpoints->begin() + i);
        tfn_changed = true;
      }
    }
    // drag color control point
    else if (ImGui::IsItemActive()) {
      ImVec2 delta = ImGui::GetIO().MouseDelta;
      if (i > 0 && i < current_colorpoints->size() - 1) {
        (*current_colorpoints)[i].position += delta.x / size.x;
        (*current_colorpoints)[i].position = clamp((*current_colorpoints)[i].position, (*current_colorpoints)[i - 1].position, (*current_colorpoints)[i + 1].position);
      }
      tfn_changed = true;
    }
  }
  return vec4f();
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__alpha_control_points(/**/
    void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &alpha_len)
{
  auto draw_list = (ImDrawList *)_draw_list;
  // draw circles
  for (int i = 0; i < current_alphapoints->size(); ++i) {
    const ImVec2 pos(cursor.x + size.x * (*current_alphapoints)[i].pos.x + margin.x, cursor.y - size.y * (*current_alphapoints)[i].pos.y - margin.z);
    ImGui::SetCursorScreenPos(ImVec2(pos.x - alpha_len, pos.y - alpha_len));
    ImGui::InvisibleButton(("##AlphaControl-" + std::to_string(i)).c_str(), ImVec2(2.f * alpha_len, 2.f * alpha_len));
    ImGui::SetCursorScreenPos(ImVec2(cursor.x, cursor.y));
    // dark bounding box
    draw_list->AddCircleFilled(pos, alpha_len, 0xFF565656);
    // white background
    draw_list->AddCircleFilled(pos, 0.8f * alpha_len, 0xFFD8D8D8);
    // highlight
    draw_list->AddCircleFilled(pos, 0.6f * alpha_len, ImGui::IsItemHovered() ? 0xFF051c33 : 0xFFD8D8D8);
    // delete alpha point
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
      if (i > 0 && i < current_alphapoints->size() - 1) {
        current_alphapoints->erase(current_alphapoints->begin() + i);
        tfn_changed = true;
      }
    } 
    // drag alpha control point
    else if (ImGui::IsItemActive()) {
      ImVec2 delta = ImGui::GetIO().MouseDelta;
      (*current_alphapoints)[i].pos.y -= delta.y / size.y;
      (*current_alphapoints)[i].pos.y = clamp((*current_alphapoints)[i].pos.y, 0.0f, 1.0f);
      if (i > 0 && i < current_alphapoints->size() - 1) {
        (*current_alphapoints)[i].pos.x += delta.x / size.x;
        (*current_alphapoints)[i].pos.x = clamp((*current_alphapoints)[i].pos.x, (*current_alphapoints)[i - 1].pos.x, (*current_alphapoints)[i + 1].pos.x);
      }
      tfn_changed = true;
    }
  }
  return vec4f();
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__alpha_freehand(
    void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing*/
    const tfn::vec2f &size,
    const tfn::vec4f &cursor)
{
  auto draw_list = (ImDrawList *)_draw_list;
  const float scroll_x = ImGui::GetScrollX();
  const float scroll_y = ImGui::GetScrollY();

  // outline the freehand curve on top of the preview fill
  {
    std::vector<ImVec2> line;
    line.reserve(draw_alphapoints.size());
    for (auto &pt : draw_alphapoints) {
      line.emplace_back(cursor.x + margin.x + pt.pos.x * size.x, cursor.y - pt.pos.y * size.y - margin.z);
    }
    if (line.size() > 1)
      draw_list->AddPolyline(line.data(), (int)line.size(), 0xFF6FD8FF, 0, 2.f);
  }

  // single interaction region covering the whole alpha canvas
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y - size.y - margin.z));
  ImGui::InvisibleButton("##tfn_alpha_freehand", ImVec2(size.x, size.y));

  if (ImGui::IsItemActive() && ImGui::IsMouseDown(0) && size.x > 0) {
    const float mouse_x = ImGui::GetMousePos().x;
    const float mouse_y = ImGui::GetMousePos().y;
    const float x = clamp((mouse_x - cursor.x - margin.x - scroll_x) / (float)size.x, 0.f, 1.f);
    const float y = clamp(-(mouse_y - cursor.y + margin.x - scroll_y) / (float)size.y, 0.f, 1.f);
    const int N = (int)draw_alphapoints.size();
    const float col_f = x * (float)(N - 1);

    if (!draw_dragging) {
      draw_dragging   = true;
      draw_last_col   = col_f;
      draw_last_alpha = y;
    }

    // paint every column between the last frame's position and this frame's,
    // so a fast stroke doesn't leave unpainted gaps
    int c0 = (int)std::round(std::min(draw_last_col, col_f));
    int c1 = (int)std::round(std::max(draw_last_col, col_f));
    c0 = (int)clamp(c0, 0, N - 1);
    c1 = (int)clamp(c1, 0, N - 1);
    for (int c = c0; c <= c1; ++c) {
      const float t = (c1 > c0) ? (float)(c - c0) / (float)(c1 - c0) : 0.f;
      const float a = (col_f >= draw_last_col) ? draw_last_alpha + t * (y - draw_last_alpha)
                                                : y + t * (draw_last_alpha - y);
      draw_alphapoints[c].pos.y = a;
    }
    draw_last_col   = col_f;
    draw_last_alpha = y;
    tfn_changed = true;
  }
  else {
    draw_dragging = false;
  }

  return vec4f();
}

inline void TransferFunctionWidget::rasterize_alpha_to_draw_curve()
{
  const int N = 256;
  draw_alphapoints.resize(N);
  for (int i = 0; i < N; ++i) {
    const float p = (float)i / (float)(N - 1);
    int il, ir;
    std::tie(il, ir) = find_interval(real_alphapoints, p);
    const float pl = real_alphapoints->at(il).pos.x;
    const float pr = real_alphapoints->at(ir).pos.x;
    const float a  = lerp(real_alphapoints->at(il).pos.y, real_alphapoints->at(ir).pos.y, pl, pr, p);
    draw_alphapoints[i].pos.x = p;
    draw_alphapoints[i].pos.y = a;
  }
}

namespace detail {
inline float point_segment_distance(const tfn::vec2f &p, const tfn::vec2f &a, const tfn::vec2f &b)
{
  const float abx = b.x - a.x, aby = b.y - a.y;
  const float len2 = abx * abx + aby * aby;
  if (len2 < 1e-12f) {
    const float dx = p.x - a.x, dy = p.y - a.y;
    return std::sqrt(dx * dx + dy * dy);
  }
  const float t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2;
  const float px = a.x + t * abx, py = a.y + t * aby;
  const float dx = p.x - px, dy = p.y - py;
  return std::sqrt(dx * dx + dy * dy);
}

// Douglas-Peucker: marks points in [lo, hi] to keep, given a max perpendicular-distance epsilon.
inline void douglas_peucker(const std::vector<tfn::vec2f> &pts, int lo, int hi, float epsilon, std::vector<bool> &keep)
{
  if (hi <= lo + 1) return;
  float max_dist = -1.f;
  int max_idx = -1;
  for (int i = lo + 1; i < hi; ++i) {
    const float d = point_segment_distance(pts[i], pts[lo], pts[hi]);
    if (d > max_dist) { max_dist = d; max_idx = i; }
  }
  if (max_dist > epsilon) {
    keep[max_idx] = true;
    douglas_peucker(pts, lo, max_idx, epsilon, keep);
    douglas_peucker(pts, max_idx, hi, epsilon, keep);
  }
}

// Linearly resample a uniformly-spaced value array to exactly n entries.
// Used when loading a raw alpha curve whose saved resolution differs from ours.
inline std::vector<float> resample_uniform(const std::vector<float> &v, int n)
{
  std::vector<float> out(n);
  if (v.empty()) { std::fill(out.begin(), out.end(), 0.f); return out; }
  if ((int)v.size() == 1) { std::fill(out.begin(), out.end(), v[0]); return out; }
  for (int i = 0; i < n; ++i) {
    const float p  = (n > 1) ? (float)i / (float)(n - 1) * (float)(v.size() - 1) : 0.f;
    const int   lo = (int)p;
    const int   hi = std::min(lo + 1, (int)v.size() - 1);
    const float t  = p - (float)lo;
    out[i] = v[lo] * (1.f - t) + v[hi] * t;
  }
  return out;
}
} // namespace detail

inline void TransferFunctionWidget::simplify_draw_curve_to_controlpoints(int max_points)
{
  const int N = (int)draw_alphapoints.size();
  if (N < 2 || !real_alphapoints || max_points < 2) return;

  std::vector<tfn::vec2f> pts(N);
  for (int i = 0; i < N; ++i) pts[i] = draw_alphapoints[i].pos;

  // binary search the largest epsilon whose simplification still fits max_points
  std::vector<bool> keep(N, false);
  float lo_eps = 0.f, hi_eps = 1.f;
  for (int iter = 0; iter < 24; ++iter) {
    const float mid = 0.5f * (lo_eps + hi_eps);
    std::fill(keep.begin(), keep.end(), false);
    keep.front() = true;
    keep.back()  = true;
    detail::douglas_peucker(pts, 0, N - 1, mid, keep);
    const int count = (int)std::count(keep.begin(), keep.end(), true);
    if (count > max_points) lo_eps = mid;
    else hi_eps = mid;
  }
  std::fill(keep.begin(), keep.end(), false);
  keep.front() = true;
  keep.back()  = true;
  detail::douglas_peucker(pts, 0, N - 1, hi_eps, keep);

  real_alphapoints->clear();
  for (int i = 0; i < N; ++i) {
    if (keep[i]) real_alphapoints->push_back(AlphaPoint(vec2f(pts[i].x, pts[i].y)));
  }
}

inline tfn::vec4f TransferFunctionWidget::draw_tfn_editor__interaction_blocks(/**/
    void *_draw_list,
    const tfn::vec3f &margin, /* left, right, spacing */
    const tfn::vec2f &size,
    const tfn::vec4f &cursor,
    const float &color_len,
    const float &alpha_len)
{
  const float mouse_x = ImGui::GetMousePos().x;
  const float mouse_y = ImGui::GetMousePos().y;
  const float scroll_x = ImGui::GetScrollX();
  const float scroll_y = ImGui::GetScrollY();
  auto draw_list = (ImDrawList *)_draw_list;
  ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y - margin.z));
  ImGui::InvisibleButton("##tfn_palette_color", ImVec2(size.x, 2.5f * color_len));
  // add color point
  if (current_tfn_editable.x && ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
    const float p = clamp((mouse_x - cursor.x - margin.x - scroll_x) / (float)size.x, 0.f, 1.f);
    int il, ir;
    std::tie(il, ir) = find_interval(current_colorpoints, p);
    const float pr = (*current_colorpoints)[ir].position;
    const float pl = (*current_colorpoints)[il].position;
    const float r = lerp((*current_colorpoints)[il].color.x, (*current_colorpoints)[ir].color.x, pl, pr, p);
    const float g = lerp((*current_colorpoints)[il].color.y, (*current_colorpoints)[ir].color.y, pl, pr, p);
    const float b = lerp((*current_colorpoints)[il].color.z, (*current_colorpoints)[ir].color.z, pl, pr, p);
    ColorPoint pt;
    pt.position = p, pt.color.x = r, pt.color.y = g, pt.color.z = b;
    current_colorpoints->insert(current_colorpoints->begin() + ir, pt);
    tfn_changed = true;
  }
  // draw background interaction (skipped in freehand draw mode -- that mode
  // owns its own interaction region so its InvisibleButton isn't shadowed by this one)
  if (!alpha_draw_mode) {
    ImGui::SetCursorScreenPos(ImVec2(cursor.x + margin.x, cursor.y - size.y - margin.z));
    if (size.x > 0 && size.y > 0) ImGui::InvisibleButton("##tfn_palette_alpha", ImVec2(size.x, size.y));
    // add alpha point
    if (current_tfn_editable.y && ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
      const float x = clamp((mouse_x - cursor.x - margin.x - scroll_x) / (float)size.x, 0.f, 1.f);
      const float y = clamp(-(mouse_y - cursor.y + margin.x - scroll_y) / (float)size.y, 0.f, 1.f);
      int il, ir;
      std::tie(il, ir) = find_interval(current_alphapoints, x);
      AlphaPoint pt;
      pt.pos.x = x, pt.pos.y = y;
      current_alphapoints->insert(current_alphapoints->begin() + ir, pt);
      tfn_changed = true;
    }
  }
  return vec4f();
}

inline void TransferFunctionWidget::draw_tfn_editor(const float margin, const float height)
{
  // style
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const float canvas_x = ImGui::GetCursorScreenPos().x;
  float canvas_y = ImGui::GetCursorScreenPos().y;
  const float width = ImGui::GetContentRegionAvail().x - 2.f * margin;
  const float color_len = 10.f;
  const float alpha_len = 10.f;
  // debug
  const tfn::vec3f m{margin, margin, margin};
  const tfn::vec2f s{width, height};
  tfn::vec4f c = {canvas_x, canvas_y, ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y};
  // draw preview texture
  c = draw_tfn_editor__preview_texture(draw_list, m, s, c);
  canvas_y = c.y;
  // draw color control points
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  if (current_tfn_editable.x) {
    draw_tfn_editor__color_control_points(draw_list, m, s, c, color_len);
  }
  // draw alpha control points (or the freehand curve, in draw mode)
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  if (alpha_draw_mode) {
    draw_tfn_editor__alpha_freehand(draw_list, m, s, c);
  } else if (current_tfn_editable.y) {
    draw_tfn_editor__alpha_control_points(draw_list, m, s, c, alpha_len);
  }
  // draw background interaction
  draw_tfn_editor__interaction_blocks(draw_list, m, s, c, color_len, alpha_len);
  // update cursors
  canvas_y += 4.f * color_len + margin;
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
}

inline bool TransferFunctionWidget::build(bool *p_open, bool do_not_render_textures)
{
  // ImGui::ShowTestWindow();

  ImGui::SetNextWindowSizeConstraints(ImVec2(400, 250), ImVec2(FLT_MAX, FLT_MAX));

  if (!ImGui::Begin("Transfer Function Widget", p_open)) {
    ImGui::End();
    return false;
  }

  build_gui();

  ImGui::End();

  if (!do_not_render_textures)
    render();

  return true;
}

inline void TransferFunctionWidget::build_gui()
{
  //------------ Styling ------------------------------
  const float margin = 10.f;

  //------------ Basic Controls -----------------------
  ImGui::Spacing();
  ImGui::SetCursorPosX(margin);
  ImGui::BeginGroup();
  {
    // /* title */
    // ImGui::Text("1D Transfer Function Editor");
    // ImGui::SameLine();
    // {
    //   ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.f);
    //   ImGui::Button("help");
    //   if (ImGui::IsItemHovered()) {
    //     ImGui::SetTooltip(
    //         "Double right click a control point to delete it\n"
    //         "Single left click and drag a control point to move it\n"
    //         "Double left click on an empty area to add a control point\n");
    //   }
    //   ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.f);
    // }
    // ImGui::Spacing();

    /* Built-in color blueprints -- click a swatch to apply its colors to the
       curve being edited now. Alpha is never touched by this. */
    draw_color_blueprint_picker();

    /* freehand draw mode for the alpha curve */
    if (current_tfn_editable.y) {
      bool prev_mode = alpha_draw_mode;
      ImGui::Checkbox("draw alpha curve", &alpha_draw_mode);
      if (alpha_draw_mode && !prev_mode) {
        rasterize_alpha_to_draw_curve();
        current_alphapoints = &draw_alphapoints;
        tfn_changed = true;
      } else if (!alpha_draw_mode && prev_mode) {
        simplify_draw_curve_to_controlpoints(8);
        current_alphapoints = real_alphapoints;
        draw_dragging = false;
        tfn_changed = true;
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "When checked, drag inside the alpha graph to freehand-paint the curve.\n"
            "Unchecking approximates the drawn curve with up to 8 control points.");
      }
    }

    /* Display transfer function value range */
    static vec2f value_range_percentage(0.f, 100.f);
    if (defaultRange.y > defaultRange.x) {
      ImGui::Text(" default value range (%.6f, %.6f)", defaultRange.x, defaultRange.y);
      ImGui::Text(" current value range (%.6f, %.6f)", valueRange.x, valueRange.y);
      if (ImGui::DragFloat2(" value range %", (float *)&value_range_percentage, 1.f, 0.f, 100.f, "%.3f")) {
        tfn_changed = true;
        valueRange.x = (defaultRange.y - defaultRange.x) * value_range_percentage.x * 0.01f + defaultRange.x;
        valueRange.y = (defaultRange.y - defaultRange.x) * value_range_percentage.y * 0.01f + defaultRange.x;
      }
    }
  }

  ImGui::EndGroup();

  //------------ Transfer Function Editor -------------

  ImGui::Spacing();
  // Fixed height so the graph doesn't resize with the window (width still
  // fills available space -- see the `width` calc inside draw_tfn_editor).
  static constexpr float kAlphaGraphHeight = 150.f;
  draw_tfn_editor(11.f, kAlphaGraphHeight);

  //------------ End Transfer Function Editor ---------
}

inline void renderTFNTexture(GLuint &tex, int width, int height)
{
  GLint prev_binding = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_binding);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  if (prev_binding) {
    glBindTexture(GL_TEXTURE_2D, prev_binding);
  }
}

inline void TransferFunctionWidget::render(int tfn_w, int tfn_h)
{
  // Upload to GL if the transfer function has changed
  if (!tfn_palette) {
    renderTFNTexture(tfn_palette, tfn_w, tfn_h);
  } else {
    /* ... */
  }

  // Update texture color
  if (tfn_changed) {
    // Backup old states
    GLint prev_binding = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_binding);

    // Sample the palette then upload the data
    std::vector<uint8_t> palette(tfn_w * tfn_h * 4, 0);
    std::vector<vec3f> colors(tfn_w, 1.f);
    std::vector<vec2f> alpha(tfn_w, 1.f);
    const float step = 1.0f / (float)(tfn_w - 1);
    for (int i = 0; i < tfn_w; ++i) {
      const float p = clamp(i * step, 0.0f, 1.0f);
      int ir, il;
      /* color */
      {
        std::tie(il, ir) = find_interval(current_colorpoints, p);
        float pl = current_colorpoints->at(il).position;
        float pr = current_colorpoints->at(ir).position;
        const float r = lerp(current_colorpoints->at(il).color.x, current_colorpoints->at(ir).color.x, pl, pr, p);
        const float g = lerp(current_colorpoints->at(il).color.y, current_colorpoints->at(ir).color.y, pl, pr, p);
        const float b = lerp(current_colorpoints->at(il).color.z, current_colorpoints->at(ir).color.z, pl, pr, p);
        colors[i].x = r;
        colors[i].y = g;
        colors[i].z = b;
        /* palette */
        palette[i * 4 + 0] = static_cast<uint8_t>(r * 255.f);
        palette[i * 4 + 1] = static_cast<uint8_t>(g * 255.f);
        palette[i * 4 + 2] = static_cast<uint8_t>(b * 255.f);
        palette[i * 4 + 3] = 255;
      }
      /* alpha */
      {
        std::tie(il, ir) = find_interval(current_alphapoints, p);
        float pl = current_alphapoints->at(il).pos.x;
        float pr = current_alphapoints->at(ir).pos.x;
        const float a = lerp(current_alphapoints->at(il).pos.y, current_alphapoints->at(ir).pos.y, pl, pr, p);
        alpha[i].x = p;
        alpha[i].y = a;
      }
    }

    // Render palette again
    glBindTexture(GL_TEXTURE_2D, tfn_palette);
    glTexImage2D(GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        tfn_w,
        tfn_h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        static_cast<const void *>(palette.data())); // We need to resize texture of texture is resized
    if (prev_binding) { // Restore previous binded texture
      glBindTexture(GL_TEXTURE_2D, prev_binding);
    }

    this->_setter_cb(colors, alpha, valueRange);
    tfn_changed = false;
  }
}

inline bool TransferFunctionWidget::load(const std::string &filename)
{
  TransferFunctionCore tfn;
  bool has_raw_alpha = false;
  std::vector<float> raw_alpha;
  try {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("could not open file");
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    json root = json::parse(text, nullptr, true, true);
    const json &jstfn = root.contains("view") ? root["view"]["volume"]["transferFunction"]
                                               : root["transferFunction"];
    loadTransferFunction(jstfn, tfn);

    // "raw" alpha mode: the alpha curve was freehand-drawn, so it is stored
    // as a dense per-column array instead of (an approximation via) control
    // points. Recover it exactly rather than relying on opacityControl.
    if (jstfn.contains("alphaMode") && jstfn["alphaMode"].get<std::string>() == "raw"
        && jstfn.contains("alphaRaw")) {
      raw_alpha = jstfn["alphaRaw"].get<std::vector<float>>();
      has_raw_alpha = !raw_alpha.empty();
    }
  }
  catch (...) {
    std::cout << "failed to load file: " << filename << std::endl;
    return false;
  }

  tfns.push_back(std::move(tfn));
  tfns_names.push_back(filename);
  select_tfn((int)tfns.size() - 1);

  if (has_raw_alpha) {
    const int N = (int)draw_alphapoints.size() > 0 ? (int)draw_alphapoints.size() : 256;
    if ((int)raw_alpha.size() != N)
      raw_alpha = detail::resample_uniform(raw_alpha, N);
    draw_alphapoints.resize(N);
    for (int i = 0; i < N; ++i) {
      draw_alphapoints[i].pos.x = (float)i / (float)(N - 1);
      draw_alphapoints[i].pos.y = clamp(raw_alpha[i], 0.f, 1.f);
    }
    alpha_draw_mode     = true;
    current_alphapoints = &draw_alphapoints;
    tfn_changed = true;
  }

  return true;
}

inline bool tfn::TransferFunctionWidget::save(const std::string &filename)
{
  auto& tfn = tfns[tfn_selection];

  json root = {{"transferFunction", {}}};
  saveTransferFunction(tfn, root["transferFunction"]);

  // saveTransferFunction() writes a base64 "alphaArray" dump, but core.h's
  // updateFromAlphaControls() bakes control points into m_rgbaTable.w, never
  // into m_alphaArray -- so this field is always all-zero dead weight for a
  // control-point-edited curve. load() never reads it back (it reconstructs
  // from opacityControl/colorControls/alphaRaw instead), so drop it.
  root["transferFunction"].erase("alphaArray");

  // In freehand draw mode, opacityControl only reflects the pre-draw control
  // points (drawing edits draw_alphapoints directly, not the control vector).
  // Save the exact per-column curve too, flagged, so load() can recover it
  // precisely instead of falling back to that stale approximation.
  if (alpha_draw_mode) {
    std::vector<float> raw_alpha(draw_alphapoints.size());
    for (size_t i = 0; i < draw_alphapoints.size(); ++i)
      raw_alpha[i] = draw_alphapoints[i].pos.y;
    root["transferFunction"]["alphaMode"] = "raw";
    root["transferFunction"]["alphaRaw"]  = raw_alpha;
  } else {
    root["transferFunction"]["alphaMode"] = "controlPoints";
  }

  std::ofstream ofs(filename, std::ofstream::out);
  if (!ofs.is_open()) return false;
  ofs << root.dump();
  ofs.close();
  return ofs.good();
}

inline void TransferFunctionWidget::set_default_tfns()
{
  for (auto &ct : _predef_color_table_) {

    tfns.emplace_back();

    auto& tfn = tfns.back();

    for (size_t i = 0; i < ct.second.size() / 4; ++i) {
      tfn.addColorControl(ct.second[i * 4], ct.second[i * 4 + 1], ct.second[i * 4 + 2], ct.second[i * 4 + 3]);
    }

    tfn.addAlphaControl(vec2f(0.00f, 0.00f));
    tfn.addAlphaControl(vec2f(0.25f, 0.25f));
    tfn.addAlphaControl(vec2f(0.50f, 0.50f));
    tfn.addAlphaControl(vec2f(0.75f, 0.75f));
    tfn.addAlphaControl(vec2f(1.00f, 1.00f));
    tfn.updateColorMap();

    tfns_names.push_back(ct.first);
  }
};

inline std::vector<vec3f> TransferFunctionWidget::sample_color_gradient(std::vector<ColorPoint> *pts, int samples) const
{
  std::vector<vec3f> out(samples);
  for (int i = 0; i < samples; ++i) {
    const float p = (samples > 1) ? (float)i / (float)(samples - 1) : 0.f;
    int il, ir;
    std::tie(il, ir) = find_interval(pts, p);
    const float pl = pts->at(il).position;
    const float pr = pts->at(ir).position;
    out[i].x = lerp(pts->at(il).color.x, pts->at(ir).color.x, pl, pr, p);
    out[i].y = lerp(pts->at(il).color.y, pts->at(ir).color.y, pl, pr, p);
    out[i].z = lerp(pts->at(il).color.z, pts->at(ir).color.z, pl, pr, p);
  }
  return out;
}

inline void TransferFunctionWidget::apply_color_blueprint(int idx)
{
  if (idx < 0 || idx >= (int)blueprint_colors.size() || !current_colorpoints) return;
  *current_colorpoints = blueprint_colors[idx]; // frozen snapshot -- never the live (possibly edited) tfns[idx]
  current_tfn_editable.x = ((int)current_colorpoints->size() > 128) ? 0 : 1;
  tfn_changed = true;
}

inline void TransferFunctionWidget::draw_color_blueprint_picker()
{
  if (num_builtin_tfns <= 0) return;

  ImGui::TextUnformatted("Color blueprints (click to apply):");

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const float swatch_w = 84.f, swatch_h = 20.f, pad = 6.f;
  const int   grad_samples = 24;
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const int   per_row = std::max(1, (int)((avail_w + pad) / (swatch_w + pad)));

  for (int i = 0; i < num_builtin_tfns; ++i) {
    if (i % per_row != 0) ImGui::SameLine(0.f, pad);

    ImGui::PushID(i);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1 = ImVec2(p0.x + swatch_w, p0.y + swatch_h);

    auto grad = sample_color_gradient(&blueprint_colors[i], grad_samples);
    const float seg_w = swatch_w / (float)grad.size();
    for (size_t s = 0; s < grad.size(); ++s) {
      const ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(grad[s].x, grad[s].y, grad[s].z, 1.f));
      draw_list->AddRectFilled(ImVec2(p0.x + s * seg_w, p0.y), ImVec2(p0.x + (s + 1) * seg_w, p1.y), col);
    }

    ImGui::InvisibleButton("##blueprint", ImVec2(swatch_w, swatch_h));
    const bool hovered = ImGui::IsItemHovered();
    draw_list->AddRect(p0, p1, hovered ? 0xFFFFFFFF : 0xFF808080, 0.f, 0, hovered ? 2.f : 1.f);

    if (hovered) {
      ImGui::SetTooltip("%s\nClick to apply these colors (alpha curve is untouched)", tfns_names[i].c_str());
    }
    if (ImGui::IsItemClicked()) {
      apply_color_blueprint(i);
    }
    ImGui::PopID();
  }
}

} // namespace tfn
