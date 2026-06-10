
# Changelog
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.2] - 7. 5. 2026

Improvements of lists.

### Changed
- Changed lists indentation, so it does not underflow boxes.


## [1.0.1] - 7. 8. 2025

plainFrame improvements, [nobg] improvements, code cleaning

### Changed
- Removed margins in plainFrame
- [nobg] now does not need \setBackground afterwards
- Moved template images to special folder
- subtitle, institution, date and author commands now can be removed


## [1.0.0] - 15. 6. 2025

Slide numbers. Cudos to ruzicsi1 for this feature.

### Added
- Slide numbers in the left sidebar.
- [nonumber] parameter for the documentclass, which removes the numbering.


## [0.9.4] - 17. 5. 2025

Added new plainFrame environment.

### Added
- New plainFrame environment, that fixes the [plain] parameter for frame environment.


## [0.9.3] - 29. 4. 2025

Added blue colour style for the presentation, change of sidebar margin.

### Added
- [blue] parameter for the class, changing the colour of the presentation.

### Changed
- Changed the right margin of the sidebar, so the text is not that close to the end of the margin.


## [0.9.2] - 29. 4. 2025

Added support for slides with orange and empty background.

### Added
- [nobg] parameter for frame, which removes the background (instead of the previous solution).
- [orangeBg] parameter for frame, which colors the background to orange. It should be used with [plain] frame parameter.


## [0.9.1] - 25. 4. 2025

Added support for frame subtitle, support for SAGELAB.

### Added
- Support for frame subtitles.
- Support for SAGE screen. New class parameter "sage" sets left sidebar to size 0.2, since SAGELAB has 5 screens connected to each other. Also, the aspect ratio changes to 20:9.


## [0.9] - 18. 3. 2025

Initial version.

### Added
- The template itself.