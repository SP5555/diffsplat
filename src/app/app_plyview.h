#pragma once
#include <string>
#include "app_base.h"

/**
 * @brief App for viewing a 3D Gaussian scene from a PLY file.
 * 
 * Supports window resizing. Rendering is not yet implemented.
 */
class AppPlyView : public AppBase
{
public:
    AppPlyView(const std::string &plyPath);

protected:
    void onStart()  override;
    void onRender() override;
    void onInput()  override;
    void onWindowResize(int newWidth, int newHeight) override;

private:

    std::string plyPath;
};
