#pragma once

#include <SDL.h>
#include <imgui\imgui.h>
#include <GL\glew.h>
#include <iostream>



class Gui {
public:
	Gui();
	~Gui();

	void renderLoop();

private:
	SDL_Window *m_window;
	SDL_GLContext m_glContext;

	bool m_exit;

};