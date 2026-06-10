%% This is the ctufit-presentation document class. It is used to produce presentations for the Czech Technical University, Faculty of Information Technology students and employees.
%%
%% This is version 1.0.2, built 7. 5. 2026.
%%
%% Get the newest version from
%% https://gitlab.fit.cvut.cz/theses-templates/FITpresentation-LaTeX/
%%
%%
%% Copyright 2025, Tomas Novacek
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%  https://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%%
%% The current maintainer of this work is Tomas Novacek.
%% Alternatively, submit bug reports to the tracker at
%% https://gitlab.fit.cvut.cz/theses-templates/FITpresentation-LaTeX/issues
%%
%%
\ProvidesClass{ctufit-presentation}[2025/06/15]

\newlength{\ctufitsidebarsize}
\newlength{\ctufitlogosize}

\DeclareOption{nobg}{
	\def\@ctufitnobackground
}

\DeclareOption{nonumber}{
    \def\@ctufitnonumber
}

\DeclareOption{orange}{
    \def\@ctufitbackgroudcolor{FITOrange}
    \def\@ctufitlion{images/template/cvut-lion-orange.png}
}

\DeclareOption{blue}{
    \def\@ctufitbackgroudcolor{FITBlue}
    \def\@ctufitlion{images/template/cvut-lion-blue.png}
}

\DeclareOption{sage}{
    \PassOptionsToClass{aspectratio=209}{beamer}
    \def\@ctufitsage
}

\DeclareOption{czech}{
	\AtBeginDocument{\selectlanguage{czech}}
	\PassOptionsToPackage{english,main=czech}{babel}
	\def\@ctufitlang{C}
    \def\@ctufitlogo{images/template/fit-logo-cs.pdf}
}
\DeclareOption{english}{
	\AtBeginDocument{\selectlanguage{english}}
	\PassOptionsToPackage{czech,main=english}{babel}
	\def\@ctufitlang{E}
    \def\@ctufitlogo{images/template/fit-logo-en.pdf}
}
\DeclareOption{slovak}{
	\AtBeginDocument{\selectlanguage{slovak}}
	\PassOptionsToPackage{english,main=slovak}{babel}
	\def\@ctufitlang{S}
    \def\@ctufitlogo{images/template/fit-logo-cs.pdf}
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ExecuteOptions{orange}
\ProcessOptions\relax

\LoadClass{beamer}[2025/02/04]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PACKAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{iftex}[2020/03/06]
\iftutex % XeLaTeX and LuaLaTeX
\else
    \errmessage{Only compilation with XeLaTeX or LuaLaTeX is allowed}
    \stop
\fi

\RequirePackage{babel}[2025/02/14]
\RequirePackage{csquotes}
\RequirePackage{lmodern}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{etoolbox} % Needed for patching
\RequirePackage{environ}

\ifdefined\@ctufitsage
    \setlength{\ctufitsidebarsize}{0.2\paperwidth}
    \setlength{\ctufitlogosize}{0.15\paperwidth}
\else
    \setlength{\ctufitsidebarsize}{0.26\paperwidth}
    \setlength{\ctufitlogosize}{0.2\paperwidth}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COLOURS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\definecolor{FITOrange}{RGB}{240,171,0}
\definecolor{FITBlue}{RGB}{0,122,194}
\usecolortheme[named=\@ctufitbackgroudcolor]{structure}
\usecolortheme{rose}
\setbeamercolor{title}{fg=white}
\setbeamercolor{author}{fg=white}
\setbeamercolor{title page}{fg=white}
\setbeamercolor{logo}{bg=\@ctufitbackgroudcolor}
\setbeamercolor{section in sidebar}{fg=white}
\setbeamercolor{sidebar left}{bg=\@ctufitbackgroudcolor, fg=white}
\setbeamercolor{sidebar left text}{fg=white}
\setbeamercolor{title in sidebar}{fg=white}
\setbeamercolor{block title}{fg=BrickRed}
\pgfdeclareimage[width=\ctufitlogosize]{fit-logo}{\@ctufitlogo}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareRobustCommand{\ctufittitle}[1]{
	\def\thectufittitle{#1}
	\title{#1}
}
\DeclareRobustCommand{\ctufitsubtitle}[1]{
	\def\thectufitsubtitle{#1}
	\subtitle{#1}
}
\DeclareRobustCommand{\ctufitauthor}[1]{
	\def\thectufitauthor{#1}
	\author{#1}
}
\DeclareRobustCommand{\ctufitdate}[1]{
	\def\thectufitdate{#1}
	\date{#1}
}
\DeclareRobustCommand{\ctufitinstitute}[1]{
	\def\thectufitinstitute{#1}
	\institute{#1}
}

\newcommand{\addIfSet}[2]{%
    \ifdefempty{#1}%
    {%
        %empty
    }{%
        \textcolor{white}{\usebeamerfont{subtitle}#1}\par
        \vspace{#2}
    }%
}

\newcommand{\plainFrameText}[1]{%
    \hspace{2em}
    \textcolor{white}{\Huge #1}
}

\NewEnviron{plainFrame}[1][]%
{%
    \begingroup%
    \hoffset=-\beamer@leftsidebar%
    \advance\textwidth\beamer@sidebarwidth%
    \advance\paperheight\baselineskip%
    \baselineskip=0pt%
    \hsize\textwidth%
    
    \begin{frame}[plain,nobg,#1]%
        \begin{columns}%
            \begin{column}{\paperwidth}%
                \vspace{-1pt}
                \vfill%
                \BODY%
                \vfill%
            \end{column}%
        \end{columns}%
    \end{frame}%
    \endgroup%
}

\newcommand{\setBackground}{%
    \ifdefined\@ctufitnobackground
        % No background image
    \else
        \usebackgroundtemplate{%
            \begin{pgfpicture}{0cm}{0cm}{10cm}{10cm}
                \pgftext[at={\pgfpoint{\paperwidth+6mm}{-4mm}}, right, bottom]{\includegraphics[width=0.3\paperwidth]{\@ctufitlion}}
            \end{pgfpicture}
        }
    \fi
}

\ctufitsubtitle{}
\ctufitauthor{}
\ctufitdate{}
\ctufitinstitute{}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outer theme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\useoutertheme[width=\ctufitsidebarsize,hideothersubsections]{sidebar}
\setbeamertemplate{headline}{}
\setbeamertemplate{footline}{}

\setbeamertemplate{sidebar left}{%
    \vspace{0.5cm}
    \hspace{0.05\beamer@sidebarwidth}
    \pgfuseimage{fit-logo}
    \par
    \vspace{1cm}
    \hspace{0.05\beamer@sidebarwidth}
    \parbox{0.87\beamer@sidebarwidth}{
        \raggedright
        \large
        \usebeamercolor{white}
        \insertframetitle
    }
    \par
    \vspace{0.3cm}
    \hspace{0.05\beamer@sidebarwidth}
    \parbox{0.87\beamer@sidebarwidth}{
        \raggedright
        \scriptsize
        \usebeamercolor{white}
        \insertframesubtitle
    }
    \ifdefined\@ctufitnonumber
        % No page number
    \else
        \vfill
        \hspace{0.05\beamer@sidebarwidth}
        \parbox{0.87\beamer@sidebarwidth}{
            \raggedright
            \small
            \usebeamercolor{white}
            \insertframenumber \ / \inserttotalframenumber
        }
        \vspace{0.5cm}
    \fi
}

\newcommand\DoColourBackground{%
    \pgfpathmoveto{\pgfpointorigin}
    \pgfpathrectanglecorners{\pgfpoint{0pt}{0pt}}{\pgfpoint{\paperwidth}{-\paperheight}}
    \pgfsetfillcolor{\@ctufitbackgroudcolor}
    \pgfusepath{fill}
}

\setbeamertemplate{title page}{%
\begin{pgfpicture}
    \pgf@relevantforpicturesizefalse
    \pgftransformxshift{-\Gm@lmargin}
    \pgftransformyshift{5.5pt}
    \pgfusepath{fill}
    \DoColourBackground

    % Logo
    \pgftext[at=\pgfpoint{.1\paperwidth}{-.15\paperheight},left]{
        \pgfuseimage{fit-logo}
    }
    \pgftext[at=\pgfpoint{.1\paperwidth}{-.57\paperheight},left, bottom]{
        \begin{minipage}{8.5cm}
            \textcolor{white}{\usebeamerfont{title}\thectufittitle}
        \end{minipage}
    }
    \pgftext[at=\pgfpoint{.1\paperwidth}{-.62\paperheight},left]{
        \textcolor{white}{\rule{2cm}{0.02cm}}
    }
    \pgftext[at=\pgfpoint{.1\paperwidth}{-.65\paperheight},left,top]{
        \begin{minipage}{9cm}
            \addIfSet{\thectufitsubtitle}{0.2cm}
            \addIfSet{\thectufitauthor}{0.8cm}
            \addIfSet{\thectufitinstitute}{0cm}
            \addIfSet{\thectufitdate}{0cm}
        \end{minipage}
    }
\end{pgfpicture}%
}
{}

\setBackground

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inner Theme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\useinnertheme{rectangles}
\setbeamertemplate{navigation symbols}{}


\makeatletter
\define@key{beamerframe}{nobg}[true]{%
    \usebackgroundtemplate{}
}

\define@key{beamerframe}{colourbg}[true]{%
   \setbeamertemplate{background canvas}{%
        \color{\@ctufitbackgroudcolor}\rule{1.5\paperwidth}{\paperheight}%
    }% 
}

\makeatother

\BeforeBeginEnvironment{frame}{\setBackground}

\setlength{\leftmargini}{1.2em}
\setlength{\leftmarginii}{1.5em}
\setlength{\leftmarginiii}{1.8em}
\setlength{\leftmarginiv}{2.1em}
\setlength{\leftmarginv}{2.5em}
\setlength{\leftmarginvi}{3.0em}