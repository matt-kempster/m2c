#-----------------------------------------------------------------
# plyparser.py
#
# PLYParser class and other utilities for simplifying programming
# parsers with PLY
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------

import warnings

class Coord(object):
    """ Coordinates of a syntactic element. Consists of:
            - File name
            - Line number
            - Lex data
            - Lex position
    """
    __slots__ = ('file', 'line', 'lexdata', 'lexpos', '__weakref__')
    def __init__(self, file, line, lexdata, lexpos):
        self.file = file
        self.line = line
        self.lexdata = lexdata
        self.lexpos = lexpos

    @property
    def column(self):
        return self.lexpos - self.lexdata.rfind('\n', 0, self.lexpos)

    def source_line(self):
        start = self.lexpos - self.column + 1
        end = self.lexdata.find('\n', start)
        if end == -1:
            end = len(self.lexdata)
        return self.lexdata[start:end].rstrip('\r')

    def __str__(self):
        return "%s:%s:%s" % (self.file, self.line, self.column)

    def __repr__(self):
        return "Coord(%s:%s:%s)" % (self.file, self.line, self.column)


class ParseError(Exception):
    def __str__(self):
        return self.args[0]


class PLYParser(object):
    def _create_opt_rule(self, rulename):
        """ Given a rule name, creates an optional ply.yacc rule
            for it. The name of the optional rule is
            <rulename>_opt
        """
        optname = rulename + '_opt'

        def optrule(self, p):
            p[0] = p[1]

        optrule.__doc__ = '%s : empty\n| %s' % (optname, rulename)
        optrule.__name__ = 'p_%s' % optname
        setattr(self.__class__, optrule.__name__, optrule)

    def _coord(self, lineno, lexpos):
        return Coord(
                file=self.clex.filename,
                line=lineno,
                lexdata=self.clex.lexer.lexdata,
                lexpos=lexpos)

    def _token_coord(self, p, token_idx):
        """Returns the coordinates for the YaccProduction object 'p' indexed
        with 'token_idx'.
        """
        return self._coord(p.lineno(token_idx), p.lexpos(token_idx))

    def _parse_error(self, msg, coord):
        raise ParseError("%s: %s" % (coord, msg), coord)


def parameterized(*params):
    """ Decorator to create parameterized rules.

    Parameterized rule methods must be named starting with 'p_' and contain
    'xxx', and their docstrings may contain 'xxx' and 'yyy'. These will be
    replaced by the given parameter tuples. For example, ``p_xxx_rule()`` with
    docstring 'xxx_rule  : yyy' when decorated with
    ``@parameterized(('id', 'ID'))`` produces ``p_id_rule()`` with the docstring
    'id_rule  : ID'. Using multiple tuples produces multiple rules.
    """
    def decorate(rule_func):
        rule_func._params = params
        return rule_func
    return decorate


def template(cls):
    """ Class decorator to generate rules from parameterized rule templates.

    See `parameterized` for more information on parameterized rules.
    """
    issued_nodoc_warning = False
    for attr_name in dir(cls):
        if attr_name.startswith('p_'):
            method = getattr(cls, attr_name)
            if hasattr(method, '_params'):
                # Remove the template method
                delattr(cls, attr_name)
                # Create parameterized rules from this method; only run this if
                # the method has a docstring. This is to address an issue when
                # pycparser's users are installed in -OO mode which strips
                # docstrings away.
                # See: https://github.com/eliben/pycparser/pull/198/ and
                #      https://github.com/eliben/pycparser/issues/197
                # for discussion.
                if method.__doc__ is not None:
                    _create_param_rules(cls, method)
                elif not issued_nodoc_warning:
                    warnings.warn(
                        'parsing methods must have __doc__ for pycparser to work properly',
                        RuntimeWarning,
                        stacklevel=2)
                    issued_nodoc_warning = True
    return cls


def _create_param_rules(cls, func):
    """ Create ply.yacc rules based on a parameterized rule function

    Generates new methods (one per each pair of parameters) based on the
    template rule function `func`, and attaches them to `cls`. The rule
    function's parameters must be accessible via its `_params` attribute.
    """
    for xxx, yyy in func._params:
        # Use the template method's body for each new method
        def param_rule(self, p):
            func(self, p)

        # Substitute in the params for the grammar rule and function name
        param_rule.__doc__ = func.__doc__.replace('xxx', xxx).replace('yyy', yyy)
        param_rule.__name__ = func.__name__.replace('xxx', xxx)

        # Attach the new method to the class
        setattr(cls, param_rule.__name__, param_rule)
