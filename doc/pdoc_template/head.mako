<%!
    from pdoc.html_helpers import minify_css
%>
<%def name="homelink()" filter="minify_css">
    .homelink {
        display: block;
        font-size: 2em;
        font-weight: bold;
        color: #555;
        padding-bottom: .5em;
        border-bottom: 1px solid silver;
    }
    .homelink:hover {
        color: inherit;
    }
    .homelink img {
        max-width:20%;
        max-height: 5em;
        margin: auto;
        margin-bottom: .3em;
    }
</%def>

<style>${homelink()}</style>
<link rel="canonical" href="https://pdoc3.github.io/pdoc/doc/${module.url()[:-len('index.html')] if module.is_package else module.url()}">
<link rel="icon" href="https://pdoc.github.io/pdoc/logo.png">
