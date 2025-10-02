set nocompatible

" Bootstrap vim-plug 
if empty(glob('~/.vim/autoload/plug.vim'))
    silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
endif
runtime autoload/plug.vim
" Plugin Definitions
call plug#begin('~/.vim/plugged')
    Plug 'tpope/vim-surround'
    " Plug 'preservim/nerdcommenter'
    Plug 'tpope/vim-commentary'
    Plug 'itchyny/lightline.vim'
    Plug 'sainnhe/gruvbox-material'
    Plug 'sheerun/vim-polyglot'
    Plug 'jasonccox/vim-wayland-clipboard'
call plug#end()
" Auto install missing plugins on startup
autocmd VimEnter * if len(filter(values(g:plugs), '!isdirectory(v:val.dir)')) | PlugInstall --sync | source $MYVIMRC | endif

" For commenting using vim-commentary
filetype plugin on
filetype plugin indent on

if has('termguicolors')
    set termguicolors
endif
" ~/.vimrc  – RGB fallback just for screen*/tmux* terms
if !has('gui_running') && &term =~# '^\%(screen\|tmux\)'
    let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
    let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
endif

set background=dark
let g:gruvbox_material_background = 'soft'
let g:gruvbox_material_better_performance = 1
" === Gruvbox autoload ===
function! s:TryGruvbox() abort
    if !empty(globpath(&rtp, 'colors/gruvbox-material.vim'))
        " File exists – safe to load the theme exactly once
        colorscheme gruvbox-material
        augroup GruvboxOnce | autocmd! | augroup END
    endif
endfunction
" Run immediately (second–later starts) and after any PlugInstall
call s:TryGruvbox()
augroup GruvboxOnce
    autocmd! User PlugLoaded ++once call s:TryGruvbox()
augroup END

" https://stackoverflow.com/questions/33380451/is-there-a-difference-between-syntax-on-and-syntax-enable-in-vimscript
if !exists("g:syntax_on")
    syntax enable
endif

set tabstop=4
set shiftwidth=4
set expandtab
set mouse=
set timeoutlen=1000
set ttimeoutlen=50

" Always use the system clipboard for yank/delete/put
set clipboard=unnamedplus,unnamed

set hlsearch
set incsearch
set smartcase

" smartundo (more granular, see https://vi.stackexchange.com/questions/2376/how-to-change-undo-granularity-in-vim)
function! s:start_delete(key)
    let l:result = a:key
    if !s:deleting
        let l:result = "\<C-G>u".l:result
    endif
    let s:deleting = 1
    return l:result
endfunction

function! s:check_undo_break(char)
    if s:deleting
        let s:deleting = 0
        call feedkeys("\<BS>\<C-G>u".a:char, 'n')
    endif
endfunction

augroup smartundo
    autocmd!
    autocmd InsertEnter * let s:deleting = 0
    autocmd InsertCharPre * call s:check_undo_break(v:char)
augroup END

inoremap <expr> <BS> <SID>start_delete("\<BS>")
inoremap <expr> <C-W> <SID>start_delete("\<C-W>")
inoremap <expr> <C-U> <SID>start_delete("\<C-U>")

" Use triple quotes for python block commenting:
augroup python_block_comments
  autocmd!
  " 1) default single-line comments with #
  autocmd FileType python setlocal commentstring=#\ %s
  " 2) in Visual mode, gc calls the block-comment function
  autocmd FileType python vnoremap <buffer> gc :call PythonBlockComment()<CR>gv
augroup END

" Global ranged function—no <SID> needed in ~/.vimrc
function! PythonBlockComment() range
  let l:start = a:firstline
  let l:end   = a:lastline
  let l:ind   = indent(l:start)
  " insert opening ''' above selection
  call append(l:start - 1, repeat(' ', l:ind) . "'''")
  " insert closing ''' below selection
  call append(l:end   + 1, repeat(' ', l:ind) . "'''")
endfunction

inoremap <End> <C-o>$
inoremap <Home> <C-o>0
vnoremap <End> $
vnoremap <Home> 0

" Center after moving to next or previous search result
nnoremap n nzzzv
nnoremap N Nzzzv

nnoremap <C-Del> dw
inoremap <C-Del> <C-o>dw
inoremap <C-BS> <C-W>

" Center cursor on screen after scrolling
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz

" Keep visual mode after indenting
:vnoremap < <gv
:vnoremap > >gv

" Detect the [Z sequence as <S-Tab>
exe 'set t_kB=' . nr2char(27) . '[Z'
" Allow indenting with tab and shift-tab
xnoremap <Tab> >gv
xnoremap <S-Tab> <gv
