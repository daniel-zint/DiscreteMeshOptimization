#include "gui.h"

#include <imgui\examples\imgui_impl_opengl3.h>
#include <imgui\examples\imgui_impl_sdl.h>

Gui::Gui() 
	: m_window(nullptr)
	, m_glContext()
	, m_exit(false)
{
	// init Gui

	// setup SDL
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cerr << "SDL Init Error: " << SDL_GetError() << std::endl;
		return;
	}

	const char *glsl_version = "#version 150";
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	SDL_DisplayMode current;
	SDL_GetCurrentDisplayMode(0, &current);
	m_window = SDL_CreateWindow("Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
	m_glContext = SDL_GL_CreateContext(m_window);
	SDL_GL_SetSwapInterval(1); // Enable vsync

	// setup GLEW
	if (glewInit() != GLEW_OK) {
		std::cerr << "Glew Initialization failed!" << std::endl;
	}

	// setup Dear ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplSDL2_InitForOpenGL(m_window, m_glContext);
	ImGui_ImplOpenGL3_Init(glsl_version);

	ImGui::StyleColorsDark();

}

Gui::~Gui() {

}

void Gui::renderLoop() {
	while (!m_exit) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			ImGui_ImplSDL2_ProcessEvent(&event);
			if (event.type == SDL_QUIT) 
				m_exit = true;
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
				m_exit = true;

			// TODO: own eventHandling here
		}

		// start dear Imgui Frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(m_window);
		ImGui::NewFrame();

		// TODO: ImGui Buttons and co

		ImGui::Render();
		SDL_GL_MakeCurrent(m_window, m_glContext);
		ImGuiIO& io = ImGui::GetIO();
		glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

	}
}