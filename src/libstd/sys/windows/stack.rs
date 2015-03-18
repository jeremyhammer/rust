// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use rt::util::report_overflow;
use core::prelude::*;
use ptr;
use mem;
use libc;
use libc::types::os::arch::extra::{LPVOID, DWORD, LONG, BOOL};

/// Setup the stack information for a thread.
///
/// Must be called from the thread that is being set up. The earlier in the stack this function is
/// called the better. Calling the function multiple times for the same thread is undefined.
#[inline(always)]
pub unsafe fn setup(_is_main: bool) {
    if SetThreadStackGuarantee(&mut 0x1000) == 0 {
        panic!("failed to reserve stack space for exception handling");
    }
    if AddVectoredExceptionHandler(0, vectored_handler) == ptr::null_mut() {
        panic!("failed to install exception handler");
    }
}

#[cfg(not(test))] // in testing, use the original libstd's version
#[lang = "stack_exhausted"]
extern fn stack_exhausted() {
    // This function is not used in windows.
}

#[no_stack_check]
extern "system" fn vectored_handler(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG {
    unsafe {
        let rec = &(*(*ExceptionInfo).ExceptionRecord);
        let code = rec.ExceptionCode;

        if code != EXCEPTION_STACK_OVERFLOW {
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // We're calling into functions with stack checks, however:
        // 1. We use stack probing and native stack overflow handling support on windows; and
        // 2. Have reserved four whole kibibytes of stack space for stack overflow handling.
        // Therefore, doing so should be A-OK.
        report_overflow();
        EXCEPTION_CONTINUE_SEARCH
    }
}

pub struct EXCEPTION_RECORD {
    pub ExceptionCode: DWORD,
    pub ExceptionFlags: DWORD,
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ExceptionAddress: LPVOID,
    pub NumberParameters: DWORD,
    pub ExceptionInformation: [LPVOID; EXCEPTION_MAXIMUM_PARAMETERS]
}

pub struct EXCEPTION_POINTERS {
    pub ExceptionRecord: *mut EXCEPTION_RECORD,
    pub ContextRecord: LPVOID
}

pub type PVECTORED_EXCEPTION_HANDLER =
    extern "system" fn(ExceptionInfo: *mut EXCEPTION_POINTERS) -> LONG;

pub type ULONG = libc::c_ulong;

const EXCEPTION_CONTINUE_SEARCH: LONG = 0;
const EXCEPTION_MAXIMUM_PARAMETERS: uint = 15;
const EXCEPTION_STACK_OVERFLOW: DWORD = 0xc00000fd;

extern "system" {
    fn AddVectoredExceptionHandler(FirstHandler: ULONG,
                                   VectoredHandler: PVECTORED_EXCEPTION_HANDLER)
                                  -> LPVOID;
    fn SetThreadStackGuarantee(StackSizeInBytes: *mut ULONG) -> BOOL;
}
