
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

/// Structure definition a location in a file.
struct Location {
  int line; ///< line number.
  int col;  ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_parentheses_open = '(',
  tok_parentheses_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',
  tok_equal = '=',
  tok_comma = ',',

  tok_eof = -1,

  tok_for = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_int = -6,

  // order
  tok_less = '<',
  tok_greater = '>',
  tok_less_equal = -7,    // <=
  tok_greater_equal = -8, // >=

  // increment/decrement operator
  tok_increment = -9,  // ++
  tok_decrement = -10, // --
};

/// Lexer goes through the stream one token at a time and keeps track the
/// location in omega string or debugging purposes.
///
/// This is a mildly modified version of the Lexer from toy tutorial in MLIR.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  explicit Lexer(std::string &&fromOmega) : buffer(std::move(fromOmega)), lastLocation({0, 0}) {
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
  std::string getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  int getValue() {
    assert(curTok == tok_int);
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
    while (bufferCurrent <= bufferEnd && *bufferCurrent && *bufferCurrent != '\n')
      ++bufferCurrent;
    if (bufferCurrent <= bufferEnd && *bufferCurrent)
      ++bufferCurrent;
    llvm::StringRef result{begin.base(), static_cast<size_t>(bufferCurrent - begin)};
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

  int peek() { return curLineBuffer.front(); }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // less equal <=
    if (lastChar == '<' && peek() == '=') {
      getNextChar();
      lastChar = Token(getNextChar());
      return tok_less_equal;
    }

    // increment ++
    if (lastChar == '+' && peek() == '+') {
      getNextChar();
      lastChar = Token(getNextChar());
      return tok_increment;
    }

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char)lastChar;

      if (identifierStr == "for")
        return tok_for;
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

      numVal = stoi(numStr);
      return tok_int;
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

  /// If the current Token is an int, this contains the value.
  int numVal = 0;

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
// iteration-statement ::= for '(' {expression}? ';' {expression}? ';' {expression}? ')' statement
//
// expression ::= assignment-expression
// assignment-expression ::= conditional-expression
//                           | unary-expression assignment-operator assignment-expression
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

class VisitorBase;

class AST {
public:
  AST(Location location) : location(std::move(location)) {}
  virtual ~AST() = default;

  const Location &loc() { return location; }

  virtual void accept(VisitorBase &b) = 0;

private:
  Location location;
};

class LoopAST;
class CallAST;

class VisitorBase {
public:
  virtual ~VisitorBase() = default;

  virtual void visit(LoopAST *loop) = 0;

  virtual void visit(CallAST *call) = 0;
};

class LoopAST : public AST {
public:
  std::vector<std::unique_ptr<AST>> block;
  int start;
  std::string stop; // TODO: I think this will have to be some sort of sum type
  int step;

  LoopAST(Location loc, std::vector<std::unique_ptr<AST>> &&block, int start, std::string stop, int step)
      : AST(std::move(loc)), block(std::move(block)), start(start), stop(std::move(stop)), step(step){};

  void accept(VisitorBase &b) override { b.visit(this); }
};

class CallAST : public AST {
public:
  const int statementNumber;

  explicit CallAST(Location loc, int statementNumber) : AST(std::move(loc)), statementNumber(statementNumber){};

  void accept(VisitorBase &b) override { b.visit(this); }
};

class DumpVisitor : public VisitorBase {
  void visit(LoopAST *loop) override {
    printf("loop[start: %d, stop: %s, step %d] {\n", loop->start, loop->stop.c_str(), loop->step);
    for (auto &statement : loop->block) {
      printf("  ");
      statement->accept(*this);
    }
  }

  void visit(CallAST *call) override { printf("call[statementNumber: %d]\n", call->statementNumber); }
};

class Program {
public:
  explicit Program(std::vector<std::unique_ptr<AST>> &&statements) : statements(std::move(statements)){};

  void dump() {
    DumpVisitor d;
    for (auto &statement : statements) {
      statement->accept(d);
    }
  }

private:
  std::vector<std::unique_ptr<AST>> statements;
};

class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Program. A program is a list of statements.
  std::unique_ptr<Program> parseProgram() {
    lexer.getNextToken(); // prime the lexer

    // Parse statements one at a time and accumulate in this vector.
    std::vector<std::unique_ptr<AST>> statements;
    while (auto s = parseStatement()) {
      statements.push_back(std::move(s));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<Program>("nothing", "at end of module");
    return std::make_unique<Program>(std::move(statements));
  }

private:
#define EXPECT_AND_CONSUME(tok, context)                                                                               \
  if (lexer.getCurToken() != tok)                                                                                      \
    return parseError<AST>(tok, context);                                                                              \
  lexer.consume(tok);

  Lexer &lexer;

  std::unique_ptr<AST> parseStatement() {
    if (lexer.getCurToken() == tok_for) {
      return parseLoop();
    } else if (lexer.getCurToken() == tok_identifier) {
      return parseCall();
    } else {
      return parseError<AST>("for loop or statement call", "parsing statement");
    }
  }

  std::unique_ptr<AST> parseCall() {
    auto loc = lexer.getLastLocation();
    auto context = "in statement call";

    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>(tok_identifier, context);
    auto statement = lexer.identifierStr;
    statement.erase(statement.begin());
    auto statementNumber = std::stoi(statement);
    lexer.consume(tok_identifier);

    // '('
    EXPECT_AND_CONSUME(tok_parentheses_open, context);

    while (lexer.getCurToken() != tok_parentheses_close) {
      // 't1'|'0'
      if (lexer.getCurToken() == tok_identifier) {
        lexer.consume(tok_identifier);
      } else if (lexer.getCurToken() == tok_int) {
        lexer.consume(tok_int);
      } else {
        return parseError<AST>("identifier or int", context);
      }

      // unless this is the end of the argument list
      if (lexer.getCurToken() == tok_comma) {
        lexer.consume(tok_comma);
      }
    }

    // ')'
    EXPECT_AND_CONSUME(tok_parentheses_close, context);

    // ';'
    EXPECT_AND_CONSUME(tok_semicolon, context);

    return std::make_unique<CallAST>(loc, statementNumber);
  }

  std::unique_ptr<AST> parseLoop() {
    auto loc = lexer.getLastLocation();
    auto context = "in loop";

    // `for`
    EXPECT_AND_CONSUME(tok_for, context);

    // `(`
    EXPECT_AND_CONSUME(tok_parentheses_open, context);

    // `t1`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>(tok_identifier, context);
    auto inductionVar = lexer.identifierStr;
    lexer.consume(tok_identifier);

    // `=`
    EXPECT_AND_CONSUME(tok_equal, context);

    // `1`
    if (lexer.getCurToken() != tok_int)
      return parseError<AST>("int", context);
    auto start = lexer.getValue();
    lexer.consume(tok_int);

    // `;`
    EXPECT_AND_CONSUME(tok_semicolon, context);

    // `t1`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>("induction var", context);
    {
      auto var = lexer.getId();
      assert(var == inductionVar && "expected induction var to reman the same between initializer and condition");
    }
    lexer.consume(tok_identifier);

    // TODO: handle other cases
    // `<=`
    EXPECT_AND_CONSUME(tok_less_equal, context);

    // `T`
    if (lexer.getCurToken() != tok_identifier) // TODO: handel other cases this could be an int
      return parseError<AST>("identifier", context);
    auto stop = lexer.getId();
    lexer.consume(tok_identifier);

    // `;`
    if (lexer.getCurToken() != tok_semicolon)
      return parseError<AST>(tok_semicolon, context);
    lexer.consume(tok_semicolon);

    // `t1`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>("induction var", context);
    {
      auto var = lexer.getId();
      assert(var == inductionVar && "expected induction var to reman the same between initializer and increment");
    }
    lexer.consume(tok_identifier);

    // `++`
    // TODO: handle steps not communicated with ++
    if (lexer.getCurToken() != tok_increment)
      return parseError<AST>("increment", context);
    int step = 1;
    lexer.consume(tok_increment);

    // `)`
    EXPECT_AND_CONSUME(tok_parentheses_close, context);

    // It's valid C to do `for(int i=0; i<10; i++) s(i);` but omega doesn't seem
    // to produce that so not worrying about it.
    // `{`
    EXPECT_AND_CONSUME(tok_bracket_open, context);

    std::vector<std::unique_ptr<AST>> block;
    while (auto s = parseStatement()) {
      block.push_back(std::move(s));
      if (lexer.getCurToken() == tok_bracket_close)
        break;
    }
    EXPECT_AND_CONSUME(tok_bracket_close, context);

    return std::make_unique<LoopAST>(loc, std::move(block), start, stop, step);
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", " << lexer.getLastLocation().col
                 << "): expected '" << expected << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
#undef EXPECT_AND_CONSUME
};

struct AVisitor : VisitorBase {

  void visit(LoopAST *loop) override {
    printf("loop [start: %d, stop: %s, step %d]\n", loop->start, loop->stop.c_str(), loop->step);
    indent++;
  }

  void visit(CallAST *call) override { printf("call"); }

private:
  int indent = 0;
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

  std::string s1 = "for(t1 = 1; t1 <= T; t1++)\n";

  std::string s2 = "  for(t1 = 1; t1 <= T; t1++) {\n"
                   "    s0(t1,0,0,0);\n"
                   "  }\n";
  auto lexer = Lexer(std::move(s2));
  auto parser = Parser(lexer);
  auto program = parser.parseProgram();
  program->dump();
  // program->dump();
}