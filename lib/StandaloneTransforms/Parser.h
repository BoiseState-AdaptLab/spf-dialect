#ifndef STANDALONE_PARSER_H
#define STANDALONE_PARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace standalone {
namespace parser {

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

  // operators
  tok_plus = '+',
  tok_minus = '-',

  tok_eof = -1,

  tok_for = -2,
  tok_if = -3,

  // primary
  tok_identifier = -4,
  tok_int = -5,

  // order
  tok_less = '<',
  tok_greater = '>',
  tok_less_equal = -6,    // <=
  tok_greater_equal = -7, // >=

  // increment/decrement operator
  tok_increment = -8, // ++
  tok_decrement = -9, // --
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
    if (lastChar == tok_plus && peek() == tok_plus) {
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
      if (identifierStr == "if")
        return tok_if;
      return tok_identifier;
    }

    // Int: [0-9]+
    if (isdigit(lastChar)) {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar));

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

class SymbolOrInt {
public:
  enum SymbolOrIntKind {
    SymbolOrInt_Symbol,
    SymbolOrInt_Int,
  };

  SymbolOrInt(SymbolOrIntKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~SymbolOrInt() = default;

  SymbolOrIntKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const SymbolOrIntKind kind;
  Location location;
};

class Symbol : public SymbolOrInt {
public:
  std::string symbol;
  int increment;

  explicit Symbol(Location loc, std::string &&symbol, int increment)
      : SymbolOrInt(SymbolOrInt_Symbol, std::move(loc)),
        symbol(std::move(symbol)), increment(increment) {}

  /// LLVM style RTTI
  static bool classof(const SymbolOrInt *c) {
    return c->getKind() == SymbolOrInt_Symbol;
  }
};

class Int : public SymbolOrInt {
public:
  int val;
  Int(Location loc, int val)
      : SymbolOrInt(SymbolOrInt_Int, std::move(loc)), val(val) {}

  /// LLVM style RTTI
  static bool classof(const SymbolOrInt *c) {
    return c->getKind() == SymbolOrInt_Int;
  }
};

class LoopAST;
class StatementCallAST;
class UFAssignmentAST;

class VisitorBase {
public:
  virtual ~VisitorBase() = default;

  virtual void visit(LoopAST *loop) = 0;

  virtual void visit(StatementCallAST *call) = 0;

  virtual void visit(UFAssignmentAST *call) = 0;
};

class LoopAST : public AST {
public:
  std::string inductionVar;
  int start;
  std::unique_ptr<Symbol> stop;
  int step;
  std::vector<std::unique_ptr<AST>> block;

  LoopAST(Location loc, std::string &&inductionVar, int start,
          std::unique_ptr<Symbol> stop, int step,
          std::vector<std::unique_ptr<AST>> &&block)
      : AST(std::move(loc)), inductionVar(std::move(inductionVar)),
        start(start), stop(std::move(stop)), step(step),
        block(std::move(block)){};

  void accept(VisitorBase &b) override { b.visit(this); }
};

class UFAssignmentAST : public AST {
public:
  std::string inductionVar;
  std::string ufName;
  std::vector<std::unique_ptr<SymbolOrInt>> args;
  explicit UFAssignmentAST(Location loc, std::string &&inductionVar,
                           std::string &&ufName,
                           std::vector<std::unique_ptr<SymbolOrInt>> &&args)
      : AST(std::move(loc)), inductionVar(std::move(inductionVar)),
        ufName(std::move(ufName)), args(std::move(args)) {}

  void accept(VisitorBase &b) override { b.visit(this); }
};

class StatementCallAST : public AST {
public:
  const int statementIndex;
  std::vector<std::unique_ptr<SymbolOrInt>> args;

  explicit StatementCallAST(Location loc, int statementIndex,
                            std::vector<std::unique_ptr<SymbolOrInt>> &&args)
      : AST(std::move(loc)), statementIndex(statementIndex),
        args(std::move(args)){};

  void accept(VisitorBase &b) override { b.visit(this); }
};

class DumpVisitor : public VisitorBase {
  int indent = 0;
  std::stringstream ss;

public:
  void visit(LoopAST *loop) override {
    ss << std::string(indent, ' ');

    ss << "loop{inductionVar:" << loop->inductionVar
       << ", start:" << loop->start
       << ", stop:" << dumpSymbolOrInt(loop->stop.get())
       << ", step:" << loop->step << ", body:[\n";
    indent += 2;
    for (auto &statement : loop->block) {
      statement->accept(*this);
    }
    indent -= 2;

    // If we have an indent assume we're in a list and print a comma. This is a
    // little hacky but the alternative was to have the visitor return a
    // templated type. That's just a lot of templates. Not worth it in this
    // case.
    ss << std::string(indent, ' ') << "]}" << (indent > 0 ? ",\n" : "\n");
  }

  void visit(StatementCallAST *call) override {
    ss << std::string(indent, ' ');

    ss << "call{statementNumber:" << call->statementIndex << ", args:[";
    int first = true;
    for (auto &symbolOrInt : call->args) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      ss << dumpSymbolOrInt(symbolOrInt.get());
    }
    ss << "]}" << (indent > 0 ? ",\n" : "\n");
  }

  void visit(UFAssignmentAST *call) override { ss << "UF assignment\n"; }

  // virtual methods can't be templated so getting the visitor to return a
  // string and build it up that way would require a lot of Type Erasure
  // nonsense. This works and is pretty easy.
  std::string output() { return ss.str(); }

private:
  std::string dumpSymbolOrInt(SymbolOrInt *symbolOrInt) {
    std::stringstream ss;
    llvm::TypeSwitch<SymbolOrInt *>(symbolOrInt)
        .Case<Symbol>([&](auto *symbol) {
          ss << "symbol{symbol:" << symbol->symbol;
          if (symbol->increment) {
            ss << ",increment:" << symbol->increment;
          }
          ss << "}";
        })
        .Case<Int>(
            [&](auto *integer) { ss << "int{val:" << integer->val << "}"; })
        .Default([&](SymbolOrInt *) {
          ss << "<unknown SymbolOrInt,kind " << symbolOrInt->getKind() << ">\n";
          exit(1);
        });
    return ss.str();
  }
};

class Program {
public:
  explicit Program(std::vector<std::unique_ptr<AST>> &&statements)
      : statements(std::move(statements)){};

  std::string dump() {
    DumpVisitor d;
    for (auto &statement : statements) {
      statement->accept(d);
    }
    return d.output();
  }

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
#define EXPECT_AND_CONSUME(tok, context)                                       \
  if (lexer.getCurToken() != tok)                                              \
    return parseError<AST>(tok, context);                                      \
  lexer.consume(tok);

  Lexer &lexer;

  std::unique_ptr<AST> parseStatement() {
    if (lexer.getCurToken() == tok_for) { // for loop
      return parseLoop();
    } else if (lexer.getCurToken() == tok_identifier) { // some sort of call
      // Here we can have a UF call or a statement call. We're expecting that
      // statements look like 's0(t1,0,t2)` and uf calls look like 't1 =
      // UFi_0()'.

      // previousIdent will either be something like `t1` in the UF case or `s0`
      // in the statement call case.
      auto previousIdent = lexer.getId();
      lexer.consume(tok_identifier);

      // determine if this looks like `t1 = ` or `s0(`. The latter is a
      // statement, the former a UF assignment.
      if (lexer.getCurToken() == tok_equal) {
        return parseUFAssignment(std::move(previousIdent));
      } else {
        return parseCall(std::move(previousIdent));
      }
    } else if (lexer.getCurToken() == tok_if) { // if statement
      // We're just throwing out 'if' statements for now. The only ones in the
      // examples that are currently using just do some error checking that
      // isn't important. TODO: There's definitely are cases
      // where an if statement is important, parse them properly.

      EXPECT_AND_CONSUME(tok_if, "if statement")
      EXPECT_AND_CONSUME(tok_parentheses_open, "if statement")
      while (lexer.getCurToken() != tok_parentheses_close) {
        lexer.consume(lexer.getCurToken());
      }
      EXPECT_AND_CONSUME(tok_parentheses_close, "if statement")
      EXPECT_AND_CONSUME(tok_bracket_open, "if statement")
      auto inside = parseStatement();
      EXPECT_AND_CONSUME(tok_bracket_close, "if statement")
      return inside;
    } else {
      return parseError<AST>("for loop or statement call", "parsing statement");
    }
  }

  std::unique_ptr<AST> parseUFAssignment(std::string &&previousIdent) {
    auto loc = lexer.getLastLocation();
    auto context = "in UF assignment";

    // `t1`
    auto inductionVar = previousIdent;

    // `=`
    EXPECT_AND_CONSUME(tok_equal, context);

    // `UFi_0`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>(tok_identifier, context);
    auto ufName = unmangleUfName(lexer.identifierStr);
    lexer.consume(tok_identifier);

    // '('
    EXPECT_AND_CONSUME(tok_parentheses_open, context);

    // 't1,0,t2,0'
    auto argsPtr = parseArgs(context);
    if (!argsPtr) {
      return parseError<AST>("argument list", context);
    }
    auto args = std::move(*argsPtr);

    // ')'
    EXPECT_AND_CONSUME(tok_parentheses_close, context);

    // ';'
    EXPECT_AND_CONSUME(tok_semicolon, context);

    return std::make_unique<UFAssignmentAST>(
        loc, std::move(inductionVar), std::move(ufName), std::move(args));
  }

  std::unique_ptr<AST> parseCall(std::string &&previousIdent) {
    auto loc = lexer.getLastLocation();
    auto context = "in statement call";

    auto statement = std::move(previousIdent);
    statement.erase(statement.begin());
    auto statementNumber = std::stoi(statement);

    // '('
    EXPECT_AND_CONSUME(tok_parentheses_open, context);

    // 't1,0,t2,0'
    auto argsPtr = parseArgs(context);
    if (!argsPtr) {
      return parseError<AST>("argument list", context);
    }
    auto args = std::move(*argsPtr);

    // ')'
    EXPECT_AND_CONSUME(tok_parentheses_close, context);

    // ';'
    EXPECT_AND_CONSUME(tok_semicolon, context);

    return std::make_unique<StatementCallAST>(loc, statementNumber,
                                              std::move(args));
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
      assert(var == inductionVar && "expected induction var to reman the same "
                                    "between initializer and condition");
    }
    lexer.consume(tok_identifier);

    // TODO: handle other cases
    // `<=`
    EXPECT_AND_CONSUME(tok_less_equal, context);

    // TODO: handle other cases, this could be an int too
    // `T`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>("identifier", context);
    auto stopSymbol = lexer.getId();
    lexer.consume(tok_identifier);

    // checking for a `- 1` or similar
    int increment = 0;
    if (lexer.getCurToken() != tok_semicolon) {
      if (lexer.getCurToken() != tok_plus && lexer.getCurToken() != tok_minus) {
        return parseError<AST>("';','-', or '+'", context);
      }

      int increment_sign;
      if (lexer.getCurToken() == tok_minus) {
        increment_sign = -1;
        lexer.consume(tok_minus);
      } else if (lexer.getCurToken() == tok_plus) {
        increment_sign = 1;
        lexer.consume(tok_plus);
      }

      if (lexer.getCurToken() != tok_int)
        return parseError<AST>("int", context);
      increment = increment_sign * lexer.getValue();
      lexer.consume(tok_int);
    }
    auto stop = std::make_unique<Symbol>(lexer.lastLocation,
                                         std::move(stopSymbol), increment);

    // `;`
    if (lexer.getCurToken() != tok_semicolon)
      return parseError<AST>(tok_semicolon, context);
    lexer.consume(tok_semicolon);

    // `t1`
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AST>("induction var", context);
    {
      auto var = lexer.getId();
      assert(var == inductionVar && "expected induction var to reman the same "
                                    "between initializer and increment");
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

    return std::make_unique<LoopAST>(loc, std::move(inductionVar), start,
                                     std::move(stop), step, std::move(block));
  }

  std::unique_ptr<std::vector<std::unique_ptr<SymbolOrInt>>>
  parseArgs(const char *context) {
    std::vector<std::unique_ptr<SymbolOrInt>> args;
    while (lexer.getCurToken() != tok_parentheses_close) {
      // 't1'|'0'
      if (lexer.getCurToken() == tok_identifier) {
        args.push_back(std::make_unique<Symbol>(lexer.getLastLocation(),
                                                lexer.getId(), 0));
        lexer.consume(tok_identifier);
      } else if (lexer.getCurToken() == tok_int) {
        args.push_back(
            std::make_unique<Int>(lexer.getLastLocation(), lexer.getValue()));
        lexer.consume(tok_int);
      } else {
        return parseError<std::vector<std::unique_ptr<SymbolOrInt>>>(
            "identifier or int", context);
      }

      // unless this is the end of the argument list
      if (lexer.getCurToken() == tok_comma) {
        lexer.consume(tok_comma);
      }
    }

    return std::make_unique<std::vector<std::unique_ptr<SymbolOrInt>>>(
        std::move(args));
  }

  // TODO: copying all these strings around is definitely bad.
  std::string unmangleUfName(std::string ufMangledName) {
    // This is all a hack, it seems like the omega rename thing in hte
    // computation API ads "_<some number>", so I'm just removing that
    // here. There's definitely some way to compose the problem so
    // that this isn't a problem, and that way would be better.
    auto underscorePos = ufMangledName.find("_");
    return ufMangledName.substr(0, underscorePos);
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
#undef EXPECT_AND_CONSUME
};

} // namespace parser
} // namespace standalone
} // namespace mlir

#endif // STANDALONE_PARSER_H