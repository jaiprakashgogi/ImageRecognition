#include <octave/oct.h>


// helloworld -> The function name seen by Octave (filename should match)
// args -> The list of arguments of type `octave_value_list`
// nargout -> Number of output arguments
// String -> The help text seen in Octave
// Return type is always `octave_value_list`
DEFUN_DLD (helloworld, args, nargout,
           "Hello World Help String")
{
  int nargin = args.length ();

  octave_stdout << "Hello World has "
                << nargin << " input arguments and "
                << nargout << " output arguments.\n";

  return octave_value_list();
}
