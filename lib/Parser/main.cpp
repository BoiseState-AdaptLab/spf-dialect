
#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>

// Adapted from
// https://gist.github.com/arslancharyev31/c48d18d8f917ffe217a0e23eb3535957.
// I added the "program" production, all others I simply modified or removed.
//
// {
//     tokens=[
//          identifier='regexp:[a-zA-Z][a-zA-Z0-9_]*'
//
//          integer-constant='regexp:\d+'
//     ]
// }
//
// program ::= statement+
//
// statement ::= expression-statement
//               | compound-statement
//               | selection-statement
//               | iteration-statement
// expression-statement ::= {expression}? ';'
// compound-statement ::= '{' {statement}* '}'
// selection-statement ::= if '(' expression ')' statement
// iteration-statement ::= for '(' {expression}? ';' {expression}? ';'
// {expression}? ')' statement
//
// expression ::= assignment-expression
// assignment-expression ::= conditional-expression
//                           | unary-expression assignment-operator
//                           assignment-expression
// conditional-expression ::= logical-or-expression
// logical-or-expression ::= logical-and-expression
// logical-and-expression ::= inclusive-or-expression
// inclusive-or-expression ::= exclusive-or-expression
// exclusive-or-expression ::= and-expression
// and-expression ::= equality-expression
// equality-expression ::= relational-expression
// relational-expression ::= shift-expression
//                           | relational-expression '<' shift-expression
//                           | relational-expression '>' shift-expression
//                           | relational-expression '<=' shift-expression
//                           | relational-expression '>=' shift-expression
// shift-expression ::= additive-expression
// additive-expression ::= multiplicative-expression
//                         | additive-expression '+' multiplicative-expression
//                         | additive-expression '-' multiplicative-expression
// multiplicative-expression ::= cast-expression
// cast-expression ::= unary-expression
// unary-expression ::= postfix-expression
//                      | postfix-expression '(' {assignment-expression}* ')'
//                      | '++' unary-expression
//                      | '--' unary-expression
//                      | unary-operator cast-expression
// postfix-expression ::= primary-expression
//                        | postfix-expression '++'
//                        | postfix-expression '--'
// primary-expression ::= identifier
//                        | constant
//
// unary-operator ::= '+'
//                    | '-'
// assignment-operator ::= '='
// constant ::= integer-constant

/// Structure definition a location in a file.
struct Location {
  int line; ///< line number.
  int col;  ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};

/// Lexer goes through the stream one token at a time and keeps track the
/// location in omega string or debugging purposes.
///
/// This is a mildly modified version of the Lexer from toy tutorial in MLIR.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  explicit Lexer(std::string &&fromOmega)
      : buffer(std::move(fromOmega)), lastLocation({0, 0}) {
    bufferCurrent = buffer.begin();
    bufferEnd = buffer.end();
  }

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

  // private:
  /// Provide one line at a time, return an empty string when reaching the end
  /// of the buffer.
  llvm::StringRef readNextLine() {
    auto begin = bufferCurrent;
    while (bufferCurrent <= bufferEnd && *bufferCurrent &&
           *bufferCurrent != '\n')
      ++bufferCurrent;
    if (bufferCurrent <= bufferEnd && *bufferCurrent)
      ++bufferCurrent;
    llvm::StringRef result{begin.base(),
                           static_cast<size_t>(bufferCurrent - begin)};
    return result;
  }

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    auto nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char)lastChar;

      if (identifierStr == "return")
        return tok_return;
      if (identifierStr == "def")
        return tok_def;
      if (identifierStr == "var")
        return tok_var;
      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  // Omega string being tokenized
  std::string buffer;

  // Iterators into buffer
  std::string::iterator bufferCurrent, bufferEnd;

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifierStr;

  /// If the current Token is a number, this contains the value.
  double numVal = 0;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef curLineBuffer = "\n";
};

int main(int argc, char *argv[]) {
  std::string s = "if (X >= 1) {\n"
                  "  for(t1 = 1; t1 <= T; t1++) {\n"
                  "    s0(t1,0,0,0);\n"
                  "    for(t3 = 1; t3 <= X-1; t3++) {\n"
                  "      s0(t1,0,t3,0);\n"
                  "      s1(t1,0,t3,1);\n"
                  "    }\n"
                  "    s1(t1,0,X,1);\n"
                  "  }\n"
                  "}\n";
  std::string s1 = "var a [];\n";
  auto sb = Lexer(std::move(s1));

  std::cout << (sb.getNextToken() == tok_var) << "\n";
  std::cout << (sb.getNextToken() == tok_identifier) << "\n";
  std::cout << (sb.getNextToken() == tok_sbracket_open) << "\n";

  // LexerBuffer(*s.begin(), *s.end(), "bla");
}