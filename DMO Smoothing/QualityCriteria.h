/**
 * Copyright (C) 2018 by Daniel Zint and Philipp Guertler
 * This file is part of Discrete Mesh Optimization DMO
 * Some rights reserved. See LICENCE.
 */

#pragma once

namespace DMO {
	enum QualityCriterium {
		MEAN_RATIO,
		AREA,
		RIGHT_ANGLE,
		JACOBIAN,
		MIN_ANGLE,
		RADIUS_RATIO,
		MAX_ANGLE
	};
}